#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AttentionPooling Multiple Alignment baseline + Caption Aux Loss (Scheme-1)
- 主任务：article <-> imgset (imgset 由 AttentionPooling 聚合 5 张图)
- 辅助任务：image <-> caption（每张图对齐它自己的 caption），CLIP-style 双向 InfoNCE
- 总损失：loss = loss_ai + LAMBDA_CAP * loss_ic

支持：
- TRAIN_SET_POLICY = "all" / "random_k"
- CAP_NEG_MODE = "all" / "no_intra_article"（推荐后者更稳）
- 按 CLIP_MODEL_LIST 循环训练
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


# ===================== 你只需要改这些参数 =====================
TRAIN_JSON_PATH = "data/captioning_dataset_5imgs_train_60.json"
TEST_JSON_PATH  = "data/captioning_dataset_5imgs_test_40.json"
IMAGE_ROOT = "data/resized"

CLIP_MODEL_NAME = "ViT-L/14@336px"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = True

TEXT_CONTEXT_LENGTH = 256
TEXT_TRUNCATE = True

SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 8
PIN_MEMORY = True

EPOCHS = 15
LR = 5e-5
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 20000
GRAD_CLIP_NORM = 1.0
PRINT_EVERY = 20

# 聚合器配置（保留你原来的变量名打印用）
TX_LAYERS = 1
TX_HEADS  = 12
TX_DROPOUT = 0.1

TRAIN_SET_POLICY = "all"   # ["all", "random_k"]

# ===== 新增：caption loss（方案1）=====
USE_CAPTION_LOSS = True
LAMBDA_CAP = 0.6           # 建议先试 0.2 / 0.5
CAP_NEG_MODE = "all"  # ["all", "no_intra_article"]

# 评测分块（N^2）
TEXT_CHUNK = 32
CAND_CHUNK = 512

REQUIRE_NUM_IMAGES = 5

CKPT_DIR = "./ckpt"
# ============================================================
def _sanitize_model_name(name: str) -> str:
    return name.replace("/", "_").replace("-", "_")


def clip_infonce_bidir(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: [N,N]，GT 是同 index
    """
    n = logits.size(0)
    targets = torch.arange(n, device=logits.device)
    loss_a = F.cross_entropy(logits, targets)
    loss_b = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_a + loss_b)

# --------------------- ImageSet pooling ---------------------
class AttentionPooling(nn.Module):
    """
    tokens: [B, N, D]
    key_padding_mask: [B, N] True=padding
    out: [B, D]
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
        return (w.unsqueeze(-1) * tokens).sum(dim=1) # [B,D]


def sample_images_with_mask(imgs5: torch.Tensor, policy: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    imgs5: [B, 5, 3, H, W]
    return imgs_out, mask
      - imgs_out: 未选中的位置填 0
      - mask: [B,5] True=padding
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
    clip_model.eval()
    set_tx.eval()

    amp_enabled = (device.type == "cuda" and use_fp16)
    autocast_ctx = torch.amp.autocast("cuda", enabled=amp_enabled) if device.type == "cuda" \
                   else torch.amp.autocast("cpu", enabled=False)

    all_t, all_i = [], []
    for batch in tqdm(dataloader, desc="Build embeddings"):
        articles: List[str] = batch["articles"]
        imgs_list: List[torch.Tensor] = batch["images"]

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
            i_flat = clip_encode_image_safe(clip_model, imgs_flat).float()
            i_flat = i_flat / (i_flat.norm(dim=-1, keepdim=True) + 1e-8)
            i_tok = i_flat.view(b, n, -1)

            imgset = set_tx(i_tok, key_padding_mask=None)
            imgset = imgset / (imgset.norm(dim=-1, keepdim=True) + 1e-8)

        all_t.append(t.cpu())
        all_i.append(imgset.cpu())

    return torch.cat(all_t, dim=0), torch.cat(all_i, dim=0)

def main():
    set_seed(SEED)
    device = torch.device(DEVICE)

    save_name = _sanitize_model_name(CLIP_MODEL_NAME)
    suffix = f"cap-{USE_CAPTION_LOSS}_lam{LAMBDA_CAP}_neg{CAP_NEG_MODE}"
    save_path = os.path.join(CKPT_DIR, f"attention_pooling_{save_name}_{suffix}.pt")

    print("\n" + "=" * 90)
    print(f"[RUN] CLIP_MODEL_NAME = {CLIP_MODEL_NAME}")
    print("=" * 90)

    clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=device, jit=False)

    if TEXT_CONTEXT_LENGTH != clip_model.context_length:
        extend_clip_context_length(clip_model, TEXT_CONTEXT_LENGTH)

    freeze_to_projection_only(clip_model)
    cast_trainable_params_to_fp32(clip_model)

    with torch.no_grad():
        embed_dim = int(clip_model.text_projection.shape[1])

    set_tx = AttentionPooling(
        dim=embed_dim,
        attn_hidden=256,
        dropout=TX_DROPOUT,
    ).to(device)

    params = [p for p in clip_model.parameters() if p.requires_grad] + list(set_tx.parameters())
    optimizer = torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)

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
    print(f"Caption loss: {USE_CAPTION_LOSS}, lambda={LAMBDA_CAP}, neg_mode={CAP_NEG_MODE}")

    amp_enabled = (device.type == "cuda" and USE_FP16)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None

    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        clip_model.train()
        set_tx.train()

        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            articles: List[str] = batch["articles"]
            imgs_list: List[torch.Tensor] = batch["images"]          # list of [5,3,H,W]
            caps_list: List[List[str]] = batch.get("captions", None) # list of list[str], each len=5

            imgs5 = torch.stack(imgs_list, dim=0).to(device)  # [B,5,3,H,W]
            b, n, c, h, w = imgs5.shape
            if n != REQUIRE_NUM_IMAGES:
                raise ValueError(f"Expect {REQUIRE_NUM_IMAGES} images, got {n}")

            imgs5_used, pad_mask = sample_images_with_mask(imgs5, TRAIN_SET_POLICY)  # pad_mask True=pad

            art_tokens = clip.tokenize(
                articles,
                context_length=clip_model.context_length,
                truncate=TEXT_TRUNCATE
            ).to(device)

            # captions tokenize（如果启用 caption loss）
            if USE_CAPTION_LOSS:
                if caps_list is None:
                    raise ValueError("USE_CAPTION_LOSS=True but batch has no 'captions'. Check your dataset/collate.")
                if len(caps_list) != b or any(len(x) != n for x in caps_list):
                    raise ValueError("captions shape mismatch with images")
                caps_flat_all = [cap.strip() for caps in caps_list for cap in caps]  # len=B*5
                cap_tokens_all = clip.tokenize(
                    caps_flat_all,
                    context_length=clip_model.context_length,
                    truncate=True
                ).to(device)
            else:
                cap_tokens_all = None

            lr_scale = cosine_with_warmup_lr(global_step, total_steps, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = LR * lr_scale

            optimizer.zero_grad(set_to_none=True)

            autocast_ctx = torch.amp.autocast("cuda", enabled=amp_enabled) if device.type == "cuda" \
                           else torch.amp.autocast("cpu", enabled=False)

            with autocast_ctx:
                # ===== article embedding =====
                t = clip_encode_text_safe(clip_model, art_tokens).float()
                t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)

                # ===== image embeddings (per-image) =====
                imgs_flat = imgs5_used.view(b * n, c, h, w)
                i_flat_all = clip_encode_image_safe(clip_model, imgs_flat).float()  # [B*5,D]
                i_flat_all = i_flat_all / (i_flat_all.norm(dim=-1, keepdim=True) + 1e-8)
                i_tok = i_flat_all.view(b, n, -1)  # [B,5,D]

                # ===== imgset embedding via attention pooling =====
                imgset = set_tx(i_tok, key_padding_mask=pad_mask)
                imgset = imgset / (imgset.norm(dim=-1, keepdim=True) + 1e-8)

                # ===== 主任务：article <-> imgset =====
                logit_scale = clip_model.logit_scale.exp().float()
                logits_ai = logit_scale * (t @ imgset.t())  # [B,B]
                loss_ai = clip_infonce_bidir(logits_ai)

                # ===== 辅助任务：image <-> caption =====
                if USE_CAPTION_LOSS:
                    c_flat_all = clip_encode_text_safe(clip_model, cap_tokens_all).float()  # [B*5,D]
                    c_flat_all = c_flat_all / (c_flat_all.norm(dim=-1, keepdim=True) + 1e-8)

                    valid_mask = (~pad_mask).reshape(-1)  # [B*5] True=valid
                    i_flat = i_flat_all[valid_mask]       # [M,D]
                    c_flat = c_flat_all[valid_mask]       # [M,D]
                    M = i_flat.shape[0]

                    logits_ic = logit_scale * (i_flat @ c_flat.t())  # [M,M]

                    if CAP_NEG_MODE == "no_intra_article":
                        # 同一篇文章内的其它 caption 不当负样本（只保留跨文章 negatives + 自己的正样本）
                        # group_id: [M], 取每个 valid item 属于 batch 内第几个文章
                        group_all = torch.arange(b, device=device).repeat_interleave(n)  # [B*5]
                        group = group_all[valid_mask]                                   # [M]
                        same_article = group[:, None].eq(group[None, :])                # [M,M]
                        eye = torch.eye(M, device=device, dtype=torch.bool)
                        allow = (~same_article) | eye
                        logits_ic = logits_ic.masked_fill(~allow, -1e9)

                    loss_ic = clip_infonce_bidir(logits_ic)
                else:
                    loss_ic = t.new_tensor(0.0)
                    M = 0

                loss = loss_ai + (LAMBDA_CAP * loss_ic if USE_CAPTION_LOSS else 0.0)

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
                if USE_CAPTION_LOSS:
                    pbar.set_postfix(
                        loss=float(loss.item()),
                        loss_ai=float(loss_ai.item()),
                        loss_ic=float(loss_ic.item()),
                        lr=float(optimizer.param_groups[0]["lr"]),
                        M=int(M),
                    )
                else:
                    pbar.set_postfix(
                        loss=float(loss.item()),
                        loss_ai=float(loss_ai.item()),
                        lr=float(optimizer.param_groups[0]["lr"]),
                    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "clip_model_name": CLIP_MODEL_NAME,
            "context_length": int(clip_model.context_length),
            "embed_dim": int(embed_dim),
            "train_set_policy": TRAIN_SET_POLICY,
            "use_caption_loss": bool(USE_CAPTION_LOSS),
            "lambda_cap": float(LAMBDA_CAP),
            "cap_neg_mode": str(CAP_NEG_MODE),
            "clip_state": clip_model.state_dict(),
            "set_tx_state": set_tx.state_dict(),
        },
        save_path
    )
    print(f"Saved: {save_path}")

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
