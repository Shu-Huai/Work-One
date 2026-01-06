#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer Multiple Alignment baseline (NewsStories ECCV'22 style baseline)

核心：
- 每条样本：文章 text + N 张图（image-set）
- 每张图先用 CLIP encode_image 得到 D 维 token（共 N 个 tokens）
- 用一个小 Transformer 在 tokens 间做 contextualization
- 输出用 pooling（mean / attention / CLS）得到 image-set embedding
- text 用 CLIP encode_text 得到 text embedding
- 用 CLIP-style 双向 InfoNCE（cross-entropy）做训练
- 测试：用 text_emb vs imgset_emb 做 N×N 检索，算 R@1/5/10、MedianRank

重要：修复 dtype mismatch
- 你如果把 text_projection / visual.proj cast 到 fp32，会导致 CLIP 原生 encode_* 出现 fp16 @ fp32
- 这里实现 clip_encode_text_safe / clip_encode_image_safe，matmul 前对齐 dtype
"""

import os
import math
import random
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import set_seed, cosine_with_warmup_lr, extend_clip_context_length, cast_trainable_params_to_fp32
from utils import clip_encode_text_safe, clip_encode_image_safe, freeze_to_projection_only
from metrics.retrieval_metrics import retrieval_metrics_multi

import clip
from dataset import CaptioningDataset, captioning_collate_fn
# 如果你的文件是 GoodNewsDataset.py：改成
# from GoodNewsDataset import CaptioningDataset, captioning_collate_fn


# ===================== 你只需要改这些参数 =====================
REQUIRE_NUM_IMAGES = 5
TRAIN_JSON_PATH = f"data/captioning_dataset_{REQUIRE_NUM_IMAGES}imgs_train_60.json"
TEST_JSON_PATH  = f"data/captioning_dataset_{REQUIRE_NUM_IMAGES}imgs_test_40.json"
IMAGE_ROOT = "data/resized"

CLIP_MODEL_NAME = "ViT-L/14@336px"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = True

TEXT_CONTEXT_LENGTH = 256
TEXT_TRUNCATE = True

SEED = 42
BATCH_SIZE = 64              # ViT-L/14@336 + ctx256 建议 16 起步
NUM_WORKERS = 8
PIN_MEMORY = True

EPOCHS = 20
LR = 5e-5                    # 贴论文：1e-5（你也可试 2e-5 / 5e-5）
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 20000         # 会自动缩短
GRAD_CLIP_NORM = 1.0
PRINT_EVERY = 20

# Transformer 聚合器配置（参考值）
TX_LAYERS = 1
TX_HEADS  = 12
TX_DROPOUT = 0.1

# ===== 新增：Pooling 策略配置 =====
# - "mean": 对 transformer 输出的 token（不含 CLS）做 masked mean pooling
# - "attn": 对 token（不含 CLS）做 attention pooling（可学习 query）
# - "cls" : 直接取 transformer 输出的 CLS token 表示
SET_POOLING = "mean"          # ["mean", "attn", "cls"]
ATTN_HIDDEN = 256             # 仅当 SET_POOLING="attn" 生效
ATTN_POOL_DROPOUT = 0.1       # 仅当 SET_POOLING="attn" 生效

# 训练时 image-set 大小策略：
# - "all": 每条都用 N 张
# - "random_k": 每条随机取 k 张（k~Uniform(1,N)），其余 padding + mask
TRAIN_SET_POLICY = "all"

# 保存（建议把 pooling 写进文件名，避免覆盖）
SAVE_NAME = CLIP_MODEL_NAME.replace("/","").replace("-","")
SAVE_PATH = f"./ckpt/transformer_multiple_{SAVE_NAME}_pool-{SET_POOLING}.pt"

# 评测分块（N^2）
EVAL_EVERY = 1
EVAL_MIN_EPOCH = 20
TEXT_CHUNK = 32
CAND_CHUNK = 512

# ===== Grid Search（新增）=====
PARAM_GRID = [
    (0.1, 4, 16),
    (0.1, 4, 1),
    (0.1, 1, 16),
    (0.1, 1, 2),
    (0.1, 2, 16),
    (0.1, 2, 12),
    (0.1, 1, 1),
    (0.1, 2, 1),
    (0.1, 2, 2),
    (0.2, 1, 1),
]

RESULT_TXT_PATH = f"./grid_results_{SAVE_NAME}_pool-{SET_POOLING}_02.txt"
# =============================
# --------------------- Pooling modules ---------------------
class AttentionPooling(nn.Module):
    """
    tokens: [B, N, D]
    key_padding_mask: [B, N] True 表示 padding（要 mask 掉）
    out: [B, D]
    """
    def __init__(self, dim: int, attn_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(dim, attn_hidden)
        self.query = nn.Parameter(torch.randn(attn_hidden))
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # [B,N,D] -> [B,N,H]
        h = torch.tanh(self.proj(tokens))
        h = self.drop(h)

        # [B,N]
        score = torch.matmul(h, self.query)

        # mask padding: True 表示 padding，需要置为 -inf
        if key_padding_mask is not None:
            score = score.masked_fill(key_padding_mask, float("-inf"))

        w = torch.softmax(score, dim=1)  # [B,N]
        out = (tokens * w.unsqueeze(-1)).sum(dim=1)  # [B,D]
        return out


# --------------------- ImageSet Transformer ---------------------
class ImageSetTransformer(nn.Module):
    """
    输入 tokens: [B, N, D]
    key_padding_mask: [B, N]  True 表示要 mask（padding）
    输出: [B, D]

    pooling:
      - "mean": masked mean over token outputs (exclude CLS)
      - "attn": attention pooling over token outputs (exclude CLS)
      - "cls" : take CLS token output
    """
    def __init__(
        self,
        dim: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        pooling: str = "mean",
        attn_hidden: int = 256,
        attn_pool_dropout: float = 0.1,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls, std=0.02)

        pooling = pooling.lower()
        if pooling not in {"mean", "attn", "cls"}:
            raise ValueError(f"Unknown pooling: {pooling}, choose from ['mean','attn','cls']")
        self.pooling = pooling

        self.attn_pool = None
        if self.pooling == "attn":
            self.attn_pool = AttentionPooling(dim, attn_hidden=attn_hidden, dropout=attn_pool_dropout)

    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, d = tokens.shape
        cls = self.cls.expand(b, 1, d)
        x = torch.cat([cls, tokens], dim=1)  # [B, 1+N, D]

        if key_padding_mask is not None:
            # 给 CLS token 加一列 False（不 mask）
            cls_mask = torch.zeros((b, 1), device=key_padding_mask.device, dtype=key_padding_mask.dtype)
            kpm = torch.cat([cls_mask, key_padding_mask], dim=1)  # [B, 1+N]
        else:
            kpm = None

        x = self.enc(x, src_key_padding_mask=kpm)  # [B, 1+N, D]

        if self.pooling == "cls":
            return x[:, 0, :]  # CLS token

        tok = x[:, 1:, :]  # [B,N,D] exclude CLS

        if self.pooling == "mean":
            if key_padding_mask is None:
                return tok.mean(dim=1)
            valid = (~key_padding_mask).unsqueeze(-1).float()  # [B,N,1]
            return (tok * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        # "attn"
        return self.attn_pool(tok, key_padding_mask=key_padding_mask)


def sample_images_with_mask(imgsN: torch.Tensor, policy: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    imgsN: [B, N, 3, H, W]
    return:
      imgs_out: [B, N, 3, H, W]（未选中的位置填 0）
      mask: [B, N] True 表示 padding（应被 transformer mask 掉）
    """
    b, n, c, h, w = imgsN.shape
    assert n == REQUIRE_NUM_IMAGES

    if policy == "all":
        mask = torch.zeros((b, n), device=imgsN.device, dtype=torch.bool)
        return imgsN, mask

    if policy == "random_k":
        out = torch.zeros_like(imgsN)
        mask = torch.ones((b, n), device=imgsN.device, dtype=torch.bool)
        for i in range(b):
            k = random.randint(1, n)
            idx = random.sample(range(n), k)
            out[i, idx] = imgsN[i, idx]
            mask[i, idx] = False
        return out, mask

    raise ValueError(f"Unknown TRAIN_SET_POLICY: {policy}")


@torch.no_grad()
def build_embeddings(
    dataloader: DataLoader,
    clip_model,
    set_tx: ImageSetTransformer,
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
        imgs_list: List[torch.Tensor] = batch["images"]  # list of [N,3,H,W]

        imgsN = torch.stack(imgs_list, dim=0).to(device)  # [B,N,3,H,W]
        b, n, c, h, w = imgsN.shape
        assert n == REQUIRE_NUM_IMAGES

        tokens = clip.tokenize(
            articles,
            context_length=clip_model.context_length,
            truncate=TEXT_TRUNCATE
        ).to(device)

        with autocast_ctx:
            # text
            t = clip_encode_text_safe(clip_model, tokens).float()
            t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)

            # images -> tokens
            imgs_flat = imgsN.view(b * n, c, h, w)
            i_tok = clip_encode_image_safe(clip_model, imgs_flat).float()  # [B*N, D]
            i_tok = i_tok / (i_tok.norm(dim=-1, keepdim=True) + 1e-8)
            i_tok = i_tok.view(b, n, -1)  # [B,N,D]

            # eval：默认用全部 N 张，不 mask
            imgset = set_tx(i_tok, key_padding_mask=None)
            imgset = imgset / (imgset.norm(dim=-1, keepdim=True) + 1e-8)

        all_t.append(t.cpu())
        all_i.append(imgset.cpu())

    return torch.cat(all_t, dim=0), torch.cat(all_i, dim=0)


def main():
    set_seed(SEED)
    device = torch.device(DEVICE)

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

    # 5) set transformer（要训练）
    set_tx = ImageSetTransformer(
        dim=embed_dim,
        nhead=TX_HEADS,
        num_layers=TX_LAYERS,
        dropout=TX_DROPOUT,
        pooling=SET_POOLING,
        attn_hidden=ATTN_HIDDEN,
        attn_pool_dropout=ATTN_POOL_DROPOUT,
    ).to(device)

    # 6) optimizer：CLIP trainable params + transformer params
    params = [p for p in clip_model.parameters() if p.requires_grad] + list(set_tx.parameters())
    optimizer = torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)

    # data
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
    print(f"Transformer: L={TX_LAYERS}, H={TX_HEADS}, dropout={TX_DROPOUT}, policy={TRAIN_SET_POLICY}, pooling={SET_POOLING}")

    amp_enabled = (device.type == "cuda" and USE_FP16)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None

    global_step = 0
    max_R1 = 0.0
    max_R5 = 0.0
    max_R10 = 0.0
    best_MedianRank = float("inf")
    for epoch in range(1, EPOCHS + 1):
        clip_model.train()
        set_tx.train()

        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            articles: List[str] = batch["articles"]
            imgs_list: List[torch.Tensor] = batch["images"]  # list of [N,3,H,W]

            imgsN = torch.stack(imgs_list, dim=0).to(device)  # [B,N,3,H,W]
            b, n, c, h, w = imgsN.shape
            if n != REQUIRE_NUM_IMAGES:
                raise ValueError(f"Expect {REQUIRE_NUM_IMAGES} images, got {n}")

            # training policy
            imgsN_used, pad_mask = sample_images_with_mask(imgsN, TRAIN_SET_POLICY)  # mask True=pad

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
                # text embedding
                t = clip_encode_text_safe(clip_model, tokens).float()
                t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)

                # image tokens
                imgs_flat = imgsN_used.view(b * n, c, h, w)
                i_tok = clip_encode_image_safe(clip_model, imgs_flat).float()  # [B*N, D]
                i_tok = i_tok / (i_tok.norm(dim=-1, keepdim=True) + 1e-8)
                i_tok = i_tok.view(b, n, -1)  # [B,N,D]

                # transformer set embedding（传入 mask）
                imgset = set_tx(i_tok, key_padding_mask=pad_mask)
                imgset = imgset / (imgset.norm(dim=-1, keepdim=True) + 1e-8)

                # CLIP-style logits
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
        if epoch >= EVAL_MIN_EPOCH and epoch % EVAL_EVERY == 0:
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
                f"\n[Epoch {epoch}] Transformer-Multiple Retrieval on CLIP {CLIP_MODEL_NAME}:\n"
                f"N={int(metrics['N'])}\n"
                f"R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}\n"
                f"MedianRank={metrics['MedianRank']:.1f}\n"
            )
            max_R1 = max(max_R1, metrics['R@1'])
            max_R5 = max(max_R5, metrics['R@5'])
            max_R10 = max(max_R10, metrics['R@10'])
            best_MedianRank = min(best_MedianRank, metrics['MedianRank'])
    # save
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(
        {
            "clip_model_name": CLIP_MODEL_NAME,
            "context_length": int(clip_model.context_length),
            "embed_dim": int(embed_dim),
            "set_pooling": SET_POOLING,
            "attn_hidden": int(ATTN_HIDDEN),
            "clip_state": clip_model.state_dict(),
            "set_tx_state": set_tx.state_dict(),
        },
        SAVE_PATH
    )

    print(f"Saved: {SAVE_PATH}")
    print(
        f"Max R@1={max_R1:.4f}  Max R@5={max_R5:.4f}  Max R@10={max_R10:.4f}\n"
        f"Max MedianRank={best_MedianRank:.1f}\n"
    )
    return {
        "max_R1": float(max_R1),
        "max_R5": float(max_R5),
        "max_R10": float(max_R10),
        "best_MedianRank": float(best_MedianRank),
    }


if __name__ == "__main__":
    os.makedirs("./ckpt", exist_ok=True)

    # 建议用 "w" 每次重跑覆盖；如果你想追加改成 "a"
    with open(RESULT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(f"CLIP={CLIP_MODEL_NAME}\n")
        f.write(f"CTX={TEXT_CONTEXT_LENGTH}, BATCH={BATCH_SIZE}, EPOCHS={EPOCHS}, LR={LR}\n")
        f.write(f"POOL={SET_POOLING}, POLICY={TRAIN_SET_POLICY}\n\n")
        f.flush()

        for (dropout, L, H) in PARAM_GRID:
            # 关键：覆盖全局 TX 参数
            TX_DROPOUT = dropout
            TX_LAYERS = L
            TX_HEADS = H

            # 关键：每组配置的 ckpt 路径不要覆盖
            SAVE_PATH = (
                f"./ckpt/transformer_multiple_{SAVE_NAME}_pool-{SET_POOLING}"
                f"_do{TX_DROPOUT}_L{TX_LAYERS}_H{TX_HEADS}.pt"
            )

            print("\n" + "=" * 80)
            print(f"[RUN] Dropout={TX_DROPOUT}, L={TX_LAYERS}, H={TX_HEADS}")
            print("=" * 80)

            try:
                res = main()  # main() 现在会 return dict
                line_param = f"Dropout={TX_DROPOUT}\tL={TX_LAYERS}\tH={TX_HEADS}\n"
                line_best = (
                    f"Max R@1={res['max_R1']:.4f}  Max R@5={res['max_R5']:.4f}  "
                    f"Max R@10={res['max_R10']:.4f}  Best MedianRank={res['best_MedianRank']:.1f}\n\n"
                )
                f.write(line_param)
                f.write(line_best)
                f.flush()

            except RuntimeError as e:
                # 可选：遇到 OOM 也记录并继续下一组（很实用，仍然算“最小修改”）
                msg = str(e).lower()
                if "out of memory" in msg or "cuda" in msg and "memory" in msg:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    f.write(f"Dropout={TX_DROPOUT}\tL={TX_LAYERS}\tH={TX_HEADS}\n")
                    f.write("OOM\n\n")
                    f.flush()
                    print("[OOM] skipped.")
                    continue
                raise

            finally:
                # 尽量释放显存，避免下一轮受影响
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"\nAll done. Results saved to: {RESULT_TXT_PATH}")

