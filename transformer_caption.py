#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer Multiple Alignment baseline + Caption Aux Loss (Scheme-1)

主任务：
- article <-> imgset（imgset = set transformer(tokens) pooling）
- CLIP-style 双向 InfoNCE

新增（方案1）：
- image <-> caption（每张图对齐它自己的 caption），batch 内做双向 InfoNCE
- 总损失：loss = loss_ai + LAMBDA_CAP * loss_ic

注意：
- 支持 TRAIN_SET_POLICY="all" / "random_k"
- 当 random_k 时，caption loss 只对有效图片位置计算（pad_mask=False 的位置）
"""

import os
import math
import random
from typing import List, Dict, Tuple
from utils import set_seed, cosine_with_warmup_lr, extend_clip_context_length, cast_trainable_params_to_fp32
from utils import clip_encode_text_safe, clip_encode_image_safe, freeze_to_projection_only
from metrics import retrieval_metrics_multi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import clip
from dataset import CaptioningDataset, captioning_collate_fn


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
BATCH_SIZE = 64
NUM_WORKERS = 8
PIN_MEMORY = True

EPOCHS = 20
LR = 5e-5
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 20000
GRAD_CLIP_NORM = 1.0
PRINT_EVERY = 20

# Transformer 聚合器配置
TX_LAYERS = 1
TX_HEADS  = 2
TX_DROPOUT = 0.1

# Pooling 策略
SET_POOLING = "attn"          # ["mean", "attn", "cls"]
ATTN_HIDDEN = 256
ATTN_POOL_DROPOUT = 0.1

# 训练时 image-set 大小策略
TRAIN_SET_POLICY = "all"      # ["all", "random_k"]

# ===== 新增：caption loss 权重（方案1核心超参）=====
LAMBDA_CAP = 0.6              # 建议先试 0.2 / 0.5

# 保存
SAVE_NAME = CLIP_MODEL_NAME.replace("/","").replace("-","")
SAVE_PATH = f"./ckpt/transformer_multiple_{SAVE_NAME}_pool-{SET_POOLING}_caploss-lam{LAMBDA_CAP}.pt"

# 评测分块（N^2）
EVAL_EVERY = 1
EVAL_MIN_EPOCH = 20
TEXT_CHUNK = 32
CAND_CHUNK = 512
# ============================================================
# --------------------- CLIP safe encoders (dtype align) --------------------
def clip_infonce_bidir(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: [N,N]，GT 是同 index
    """
    n = logits.size(0)
    targets = torch.arange(n, device=logits.device)
    loss_a = F.cross_entropy(logits, targets)
    loss_b = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_a + loss_b)


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
        h = torch.tanh(self.proj(tokens))   # [B,N,H]
        h = self.drop(h)
        score = torch.matmul(h, self.query) # [B,N]
        if key_padding_mask is not None:
            score = score.masked_fill(key_padding_mask, float("-inf"))
        w = torch.softmax(score, dim=1)     # [B,N]
        return (tokens * w.unsqueeze(-1)).sum(dim=1)


class ImageSetTransformer(nn.Module):
    """
    tokens: [B, N, D]
    key_padding_mask: [B, N] True = padding
    return: [B, D]
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
            raise ValueError(f"Unknown pooling: {pooling}")
        self.pooling = pooling

        self.attn_pool = None
        if self.pooling == "attn":
            self.attn_pool = AttentionPooling(dim, attn_hidden=attn_hidden, dropout=attn_pool_dropout)

    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, d = tokens.shape
        cls = self.cls.expand(b, 1, d)
        x = torch.cat([cls, tokens], dim=1)  # [B, 1+N, D]

        if key_padding_mask is not None:
            cls_mask = torch.zeros((b, 1), device=key_padding_mask.device, dtype=key_padding_mask.dtype)
            kpm = torch.cat([cls_mask, key_padding_mask], dim=1)  # [B, 1+N]
        else:
            kpm = None

        x = self.enc(x, src_key_padding_mask=kpm)  # [B,1+N,D]

        if self.pooling == "cls":
            return x[:, 0, :]

        tok = x[:, 1:, :]  # [B,N,D]

        if self.pooling == "mean":
            if key_padding_mask is None:
                return tok.mean(dim=1)
            valid = (~key_padding_mask).unsqueeze(-1).float()
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
            t = clip_encode_text_safe(clip_model, tokens).float()
            t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)

            imgs_flat = imgsN.view(b * n, c, h, w)
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
    print(f"Caption aux loss: lambda={LAMBDA_CAP}")

    amp_enabled = (device.type == "cuda" and USE_FP16)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None

    global_step = 0
    max_R1 = 0.0
    max_R5 = 0.0
    max_R10 = 0.0
    max_MedianRank = 0.0
    for epoch in range(1, EPOCHS + 1):
        clip_model.train()
        set_tx.train()

        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            articles: List[str] = batch["articles"]
            imgs_list: List[torch.Tensor] = batch["images"]      # list of [N,3,H,W]
            caps_list: List[List[str]] = batch["captions"]       # list of list[str], each len=N

            imgsN = torch.stack(imgs_list, dim=0).to(device)     # [B,N,3,H,W]
            b, n, c, h, w = imgsN.shape
            if n != REQUIRE_NUM_IMAGES:
                raise ValueError(f"Expect {REQUIRE_NUM_IMAGES} images, got {n}")
            if len(caps_list) != b or any(len(x) != n for x in caps_list):
                raise ValueError("captions shape mismatch with images")

            # training policy
            imgsN_used, pad_mask = sample_images_with_mask(imgsN, TRAIN_SET_POLICY)  # pad_mask True=pad

            # tokenize article
            art_tokens = clip.tokenize(
                articles,
                context_length=clip_model.context_length,
                truncate=TEXT_TRUNCATE
            ).to(device)

            # tokenize captions（先 flatten B*N）
            caps_flat_all = [cap.strip() for caps in caps_list for cap in caps]  # len=B*N
            cap_tokens_all = clip.tokenize(
                caps_flat_all,
                context_length=clip_model.context_length,
                truncate=True
            ).to(device)

            # lr schedule
            lr_scale = cosine_with_warmup_lr(global_step, total_steps, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = LR * lr_scale

            optimizer.zero_grad(set_to_none=True)

            autocast_ctx = torch.amp.autocast("cuda", enabled=amp_enabled) if device.type == "cuda" \
                           else torch.amp.autocast("cpu", enabled=False)

            with autocast_ctx:
                # ===== text embedding (article) =====
                t = clip_encode_text_safe(clip_model, art_tokens).float()   # [B,D]
                t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)

                # ===== image embeddings (per-image tokens) =====
                imgs_flat = imgsN_used.view(b * n, c, h, w)
                i_flat_all = clip_encode_image_safe(clip_model, imgs_flat).float()  # [B*N,D]
                i_flat_all = i_flat_all / (i_flat_all.norm(dim=-1, keepdim=True) + 1e-8)
                i_tok = i_flat_all.view(b, n, -1)  # [B,N,D]

                # ===== set transformer =====
                imgset = set_tx(i_tok, key_padding_mask=pad_mask)
                imgset = imgset / (imgset.norm(dim=-1, keepdim=True) + 1e-8)

                # ===== 主任务：article <-> imgset =====
                logit_scale = clip_model.logit_scale.exp().float()
                logits_ai = logit_scale * (t @ imgset.t())   # [B,B]
                loss_ai = clip_infonce_bidir(logits_ai)

                # ===== 辅助任务：image <-> caption（只对有效位置计算）=====
                c_flat_all = clip_encode_text_safe(clip_model, cap_tokens_all).float()  # [B*N,D]
                c_flat_all = c_flat_all / (c_flat_all.norm(dim=-1, keepdim=True) + 1e-8)

                valid_mask = (~pad_mask).reshape(-1)  # [B*N] True=valid
                i_flat = i_flat_all[valid_mask]       # [M,D]
                c_flat = c_flat_all[valid_mask]       # [M,D]
                M = i_flat.shape[0]

                # random_k 下 M 可能变化，但一定 >= B（每条至少1张）
                logits_ic = logit_scale * (i_flat @ c_flat.t())  # [M,M]
                loss_ic = clip_infonce_bidir(logits_ic)

                loss = loss_ai + LAMBDA_CAP * loss_ic

            # backward/step
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
                pbar.set_postfix(
                    loss=float(loss.item()),
                    loss_ai=float(loss_ai.item()),
                    loss_ic=float(loss_ic.item()),
                    lr=float(optimizer.param_groups[0]["lr"]),
                    M=int(M),
                )
        if epoch % EVAL_EVERY == 0 and epoch >= EVAL_MIN_EPOCH:
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
            max_MedianRank = max(max_MedianRank, metrics['MedianRank'])
    # save
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(
        {
            "clip_model_name": CLIP_MODEL_NAME,
            "context_length": int(clip_model.context_length),
            "embed_dim": int(embed_dim),
            "set_pooling": SET_POOLING,
            "attn_hidden": int(ATTN_HIDDEN),
            "lambda_cap": float(LAMBDA_CAP),
            "train_set_policy": TRAIN_SET_POLICY,
            "clip_state": clip_model.state_dict(),
            "set_tx_state": set_tx.state_dict(),
        },
        SAVE_PATH
    )
    print(f"Saved: {SAVE_PATH}")
    print(
        f"Max R@1={max_R1:.4f}  Max R@5={max_R5:.4f}  Max R@10={max_R10:.4f}\n"
        f"Max MedianRank={max_MedianRank:.1f}\n"
    )



if __name__ == "__main__":
    main()
