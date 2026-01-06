#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer Multiple Alignment baseline (NewsStories ECCV'22 style baseline)

核心：
- 每条样本：文章 text + 5 张图（image-set）
- 每张图先用 CLIP encode_image 得到 D 维 token（共 N=5 个 tokens）
- 用一个小聚合器（AttentionPooling）在 tokens 间做 contextualization
- 输出用 pooling 得到 image-set embedding
- text 用 CLIP encode_text 得到 text embedding
- 用 CLIP-style 双向 InfoNCE（cross-entropy）做训练
- 测试：用 text_emb vs imgset_emb 做 N×N 检索，算 R@1/5/10、MedianRank

重要：修复 dtype mismatch
- 你如果把 text_projection / visual.proj cast 到 fp32，会导致 CLIP 原生 encode_* 出现 fp16 @ fp32
- 这里实现 clip_encode_text_safe / clip_encode_image_safe（ViT 视觉塔），matmul 前对齐 dtype

本脚本新增：按模型列表循环训练
for item in ["ViT-L/14", "ViT-L/14@336px", "ViT-B/16","ViT-B/32","RN50x64","RN50x16","RN50x4","RN101","RN50"]
"""

import os
import math
import random
from typing import List, Dict, Tuple
from utils import set_seed, cosine_with_warmup_lr, extend_clip_context_length, cast_trainable_params_to_fp32
from utils import clip_encode_text_safe, clip_encode_image_safe, freeze_to_projection_only
from metrics.retrieval_metrics import retrieval_metrics_multi

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import clip
from dataset import CaptioningDataset, captioning_collate_fn
# 如果你的文件是 GoodNewsDataset.py：改成
# from GoodNewsDataset import CaptioningDataset, captioning_collate_fn


# ===================== 你只需要改这些参数 =====================
TRAIN_JSON_PATH = "data/captioning_dataset_5imgs_train_60.json"
TEST_JSON_PATH  = "data/captioning_dataset_5imgs_test_40.json"
IMAGE_ROOT = "data/resized"

# 你要跑的模型列表
CLIP_MODEL_NAME = "ViT-L/14@336px"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = True

TEXT_CONTEXT_LENGTH = 256
TEXT_TRUNCATE = True

SEED = 42
BATCH_SIZE = 32              # ViT-L/14@336 + ctx256 建议 16 起步
NUM_WORKERS = 8
PIN_MEMORY = True

EPOCHS = 15
LR = 5e-5                    # 贴论文：1e-5（你也可试 2e-5 / 5e-5）
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 20000         # 会自动缩短
GRAD_CLIP_NORM = 1.0
PRINT_EVERY = 20

# 聚合器配置（保留你原来的变量名打印用）
TX_LAYERS = 1
TX_HEADS  = 12
TX_DROPOUT = 0.1

# 训练时 image-set 大小策略：
# - "all": 每条都用 5 张
# - "random_k": 每条随机取 k 张（k~Uniform(1,5)），其余 padding + mask
TRAIN_SET_POLICY = "all"

# 评测分块（N^2）
TEXT_CHUNK = 32
CAND_CHUNK = 512

REQUIRE_NUM_IMAGES = 5

# 保存目录
CKPT_DIR = "./ckpt"
# ============================================================

def _sanitize_model_name(name: str) -> str:
    return (
        name.replace("/", "_")
            .replace("-", "_")
    )



# --------------------- ImageSet pooling ---------------------
class AttentionPooling(nn.Module):
    """
    输入 tokens: [B, N, D]
    key_padding_mask: [B, N] True 表示 padding（要 mask 掉）
    输出: [B, D]（加权求和）
    """
    def __init__(self, dim: int, attn_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(dim, attn_hidden)
        self.query = nn.Parameter(torch.randn(attn_hidden))
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = torch.tanh(self.proj(tokens))            # [B,N,H]
        h = self.drop(h)
        score = torch.matmul(h, self.query)          # [B,N]
        if key_padding_mask is not None:
            score = score.masked_fill(key_padding_mask, float("-inf"))
        w = torch.softmax(score, dim=1)              # [B,N]
        out = (w.unsqueeze(-1) * tokens).sum(dim=1)  # [B,D]
        return out


def sample_images_with_mask(imgs5: torch.Tensor, policy: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    imgs5: [B, 5, 3, H, W]
    return:
      imgs_out: [B, 5, 3, H, W]（未选中的位置填 0）
      mask: [B, 5] True 表示 padding（应被 mask 掉）
    """
    b, n, c, h, w = imgs5.shape
    assert n == REQUIRE_NUM_IMAGES

    if policy == "all":
        mask = torch.zeros((b, n), device=imgs5.device, dtype=torch.bool)
        return imgs5, mask

    if policy == "random_k":
        out = torch.zeros_like(imgs5)
        mask = torch.ones((b, n), device=imgs5.device, dtype=torch.bool)
        for i in range(b):
            k = random.randint(1, n)
            idx = random.sample(range(n), k)
            out[i, idx] = imgs5[i, idx]
            mask[i, idx] = False
        return out, mask

    raise ValueError(f"Unknown TRAIN_SET_POLICY: {policy}")


@torch.no_grad()
def build_embeddings(
    dataloader: DataLoader,
    clip_model,
    set_tx: AttentionPooling,
    device: torch.device,
    use_fp16: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    输出 CPU embeddings：
      text_embs  : [N, D]
      imgset_embs: [N, D]
    """
    clip_model.eval()
    set_tx.eval()

    amp_enabled = (device.type == "cuda" and use_fp16)
    autocast_ctx = torch.amp.autocast("cuda", enabled=amp_enabled) if device.type == "cuda" \
                   else torch.amp.autocast("cpu", enabled=False)

    all_t, all_i = [], []
    for batch in tqdm(dataloader, desc="Build embeddings"):
        articles: List[str] = batch["articles"]
        imgs_list: List[torch.Tensor] = batch["images"]  # list of [5,3,H,W]

        imgs5 = torch.stack(imgs_list, dim=0).to(device)  # [B,5,3,H,W]
        b, n, c, h, w = imgs5.shape
        assert n == REQUIRE_NUM_IMAGES

        tokens = clip.tokenize(
            articles,
            context_length=clip_model.context_length,
            truncate=TEXT_TRUNCATE
        ).to(device)

        with autocast_ctx:
            t = clip_encode_text_safe(clip_model, tokens).float()
            t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)

            imgs_flat = imgs5.view(b * n, c, h, w)
            i_tok = clip_encode_image_safe(clip_model, imgs_flat).float()  # [B*5, D]
            i_tok = i_tok / (i_tok.norm(dim=-1, keepdim=True) + 1e-8)
            i_tok = i_tok.view(b, n, -1)  # [B,5,D]

            imgset = set_tx(i_tok, key_padding_mask=None)
            imgset = imgset / (imgset.norm(dim=-1, keepdim=True) + 1e-8)

        all_t.append(t.cpu())
        all_i.append(imgset.cpu())

    return torch.cat(all_t, dim=0), torch.cat(all_i, dim=0)

def main():
    set_seed(SEED)
    device = torch.device(DEVICE)

    save_name = _sanitize_model_name(CLIP_MODEL_NAME)
    save_path = os.path.join(CKPT_DIR, f"attention_pooling_{save_name}.pt")

    print("\n" + "=" * 90)
    print(f"[RUN] CLIP_MODEL_NAME = {CLIP_MODEL_NAME}")
    print("=" * 90)

    # 1) load CLIP
    clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=device, jit=False)

    # 2) extend ctx
    if TEXT_CONTEXT_LENGTH != clip_model.context_length:
        extend_clip_context_length(clip_model, TEXT_CONTEXT_LENGTH)

    # 3) freeze CLIP backbone (projection-only) + fp32 trainable
    freeze_to_projection_only(clip_model)
    cast_trainable_params_to_fp32(clip_model)

    # 4) embed dim
    with torch.no_grad():
        embed_dim = int(clip_model.text_projection.shape[1])

    # 5) pooling set module（要训练）
    set_tx = AttentionPooling(
        dim=embed_dim,
        attn_hidden=256,
        dropout=TX_DROPOUT,
    ).to(device)

    # 6) optimizer：CLIP trainable params + pooling params
    params = [p for p in clip_model.parameters() if p.requires_grad] + list(set_tx.parameters())
    optimizer = torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)

    # data：preprocess 跟模型走（重要）
    train_ds = CaptioningDataset(TRAIN_JSON_PATH, IMAGE_ROOT, transform=preprocess, use_headline=False)
    test_ds  = CaptioningDataset(TEST_JSON_PATH,  IMAGE_ROOT, transform=preprocess, use_headline=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY if device.type == "cuda" else False,
        collate_fn=captioning_collate_fn,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY if device.type == "cuda" else False,
        collate_fn=captioning_collate_fn,
        drop_last=False,
    )

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = min(WARMUP_STEPS, max(0, total_steps // 10))
    print(f"Train size={len(train_ds)}, Test size={len(test_ds)}")
    print(f"Total steps={total_steps}, Warmup steps={warmup_steps}")
    print(f"CLIP ctx={clip_model.context_length}, embed_dim={embed_dim}")
    print(f"Aggregator=AttentionPooling(dropout={TX_DROPOUT}), policy={TRAIN_SET_POLICY}")
    print(f"Transformer: L={TX_LAYERS}, H={TX_HEADS}  (仅打印占位，当前实现用 AttentionPooling)")

    amp_enabled = (device.type == "cuda" and USE_FP16)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None

    global_step = 0
    last_epoch = 0
    for epoch in range(1, EPOCHS + 1):
        last_epoch = epoch
        clip_model.train()
        set_tx.train()

        pbar = tqdm(train_loader, desc=f"[Train epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            articles: List[str] = batch["articles"]
            imgs_list: List[torch.Tensor] = batch["images"]  # list of [5,3,H,W]

            imgs5 = torch.stack(imgs_list, dim=0).to(device)  # [B,5,3,H,W]
            b, n, c, h, w = imgs5.shape
            if n != REQUIRE_NUM_IMAGES:
                raise ValueError(f"Expect {REQUIRE_NUM_IMAGES} images, got {n}")

            imgs5_used, pad_mask = sample_images_with_mask(imgs5, TRAIN_SET_POLICY)

            tokens = clip.tokenize(
                articles,
                context_length=clip_model.context_length,
                truncate=TEXT_TRUNCATE
            ).to(device)

            lr_scale = cosine_with_warmup_lr(global_step, total_steps, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = LR * lr_scale

            optimizer.zero_grad(set_to_none=True)

            autocast_ctx = torch.amp.autocast("cuda", enabled=amp_enabled) if device.type == "cuda" \
                           else torch.amp.autocast("cpu", enabled=False)

            with autocast_ctx:
                t = clip_encode_text_safe(clip_model, tokens).float()
                t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)

                imgs_flat = imgs5_used.view(b * n, c, h, w)
                i_tok = clip_encode_image_safe(clip_model, imgs_flat).float()  # [B*5, D]
                i_tok = i_tok / (i_tok.norm(dim=-1, keepdim=True) + 1e-8)
                i_tok = i_tok.view(b, n, -1)  # [B,5,D]

                imgset = set_tx(i_tok, key_padding_mask=pad_mask)
                imgset = imgset / (imgset.norm(dim=-1, keepdim=True) + 1e-8)

                logit_scale = clip_model.logit_scale.exp().float()
                logits = logit_scale * (t @ imgset.t())  # [B,B]

                targets = torch.arange(b, device=logits.device)
                loss_t = F.cross_entropy(logits, targets)
                loss_i = F.cross_entropy(logits.t(), targets)
                loss = 0.5 * (loss_t + loss_i)

            if device.type == "cuda" and scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
                optimizer.step()

            global_step += 1
            if global_step % PRINT_EVERY == 0:
                pbar.set_postfix(loss=float(loss.item()), lr=float(optimizer.param_groups[0]["lr"]))

    # save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "clip_model_name": CLIP_MODEL_NAME,
            "context_length": int(clip_model.context_length),
            "embed_dim": int(embed_dim),
            "clip_state": clip_model.state_dict(),
            "set_tx_state": set_tx.state_dict(),
        },
        save_path
    )
    print(f"Saved: {save_path}")

    # eval
    clip_model.eval()
    set_tx.eval()
    with torch.no_grad():
        text_embs, imgset_embs = build_embeddings(test_loader, clip_model, set_tx, device, USE_FP16)
        metrics = retrieval_metrics_multi(
            text_embs, imgset_embs,
            logit_scale=clip_model.logit_scale,
            device=device,
            text_chunk=TEXT_CHUNK,
            cand_chunk=CAND_CHUNK,
        )
    print(
        f"\n[{CLIP_MODEL_NAME}] Retrieval:\n"
        f"N={int(metrics['N'])}\n"
        f"R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}\n"
        f"MedianRank={metrics['MedianRank']:.1f}\n"
    )

    # 清理显存，避免下一个模型 OOM
    del clip_model, set_tx, optimizer, train_loader, test_loader, train_ds, test_ds, text_embs, imgset_embs
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()