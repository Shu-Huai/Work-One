#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mean Multiple Alignment baseline (NewsStories/GoodNews style)

核心：
- 每条样本：文章 text + 5 张图（image-set）
- 每张图用 CLIP encode_image 得到 D 维 embedding
- Mean Multiple：对 5 个 image embeddings 做 mean pooling 得到 image-set embedding
- text 用 CLIP encode_text 得到 text embedding
- 用 CLIP-style 双向 InfoNCE（cross-entropy）训练
- 测试：用 text_emb vs imgset_emb 做 N×N 检索，算 R@1/5/10、MedianRank

重要：修复 dtype mismatch
- 如果你把 text_projection / visual.proj cast 到 fp32，会导致 CLIP 原生 encode_* 出现 fp16 @ fp32
- 这里实现 clip_encode_text_safe / clip_encode_image_safe，matmul 前对齐 dtype
"""

import os
import math
import random
from typing import List, Dict, Tuple

import torch
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
REQUIRE_NUM_IMAGES = 4

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

EPOCHS = 10
LR = 5e-5
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 20000
GRAD_CLIP_NORM = 1.0
PRINT_EVERY = 20

# Mean Multiple pooling 方式
# - "mean_normed": 先对每张图 L2 norm，再 mean，再 L2 norm（推荐）
# - "mean_raw":    先 mean，再 L2 norm
POOL_MODE = "mean_normed"

# 保存
SAVE_NAME = CLIP_MODEL_NAME.replace("/", "").replace("-", "")
SAVE_PATH = f"./ckpt/mean_multiple_{SAVE_NAME}.pt"

# 评测分块（N^2）
EVAL_EVERY = 1
TEXT_CHUNK = 32
CAND_CHUNK = 512

# ============================================================

def mean_multiple_pool(img_tokens: torch.Tensor, mode: str) -> torch.Tensor:
    """
    img_tokens: [B, N, D]
    return: [B, D]
    """
    if mode == "mean_normed":
        tok = img_tokens / (img_tokens.norm(dim=-1, keepdim=True) + 1e-8)
        out = tok.mean(dim=1)
        out = out / (out.norm(dim=-1, keepdim=True) + 1e-8)
        return out
    if mode == "mean_raw":
        out = img_tokens.mean(dim=1)
        out = out / (out.norm(dim=-1, keepdim=True) + 1e-8)
        return out
    raise ValueError(f"Unknown POOL_MODE: {mode}")


@torch.no_grad()
def build_embeddings(
    dataloader: DataLoader,
    clip_model,
    device: torch.device,
    use_fp16: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    输出 CPU embeddings：
      text_embs  : [N, D]
      imgset_embs: [N, D]
    """
    clip_model.eval()

    amp_enabled = (device.type == "cuda" and use_fp16)
    autocast_ctx = torch.amp.autocast("cuda", enabled=amp_enabled) if device.type == "cuda" \
                   else torch.amp.autocast("cpu", enabled=False)

    all_t, all_i = [], []
    for batch in tqdm(dataloader, desc="Build embeddings"):
        articles: List[str] = batch["articles"]
        imgs_list: List[torch.Tensor] = batch["images"]  # list of [5,3,H,W]

        imgs5 = torch.stack(imgs_list, dim=0).to(device)  # [B,5,3,H,W]
        b, n, c, h, w = imgs5.shape
        if n != REQUIRE_NUM_IMAGES:
            raise ValueError(f"Expect {REQUIRE_NUM_IMAGES} images, got {n}")

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
            imgs_flat = imgs5.view(b * n, c, h, w)
            i_tok = clip_encode_image_safe(clip_model, imgs_flat).float()  # [B*5, D]
            i_tok = i_tok.view(b, n, -1)  # [B,5,D]

            # mean multiple pooling
            imgset = mean_multiple_pool(i_tok, POOL_MODE)

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

    # 4) optimizer：CLIP trainable params
    params = [p for p in clip_model.parameters() if p.requires_grad]
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
    with torch.no_grad():
        embed_dim = int(clip_model.text_projection.shape[1])
    print(f"CLIP ctx={clip_model.context_length}, embed_dim={embed_dim}")
    print(f"Mean Multiple pooling: {POOL_MODE}")

    amp_enabled = (device.type == "cuda" and USE_FP16)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None

    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        clip_model.train()

        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            articles: List[str] = batch["articles"]
            imgs_list: List[torch.Tensor] = batch["images"]  # list of [5,3,H,W]

            imgs5 = torch.stack(imgs_list, dim=0).to(device)  # [B,5,3,H,W]
            b, n, c, h, w = imgs5.shape
            if n != REQUIRE_NUM_IMAGES:
                raise ValueError(f"Expect {REQUIRE_NUM_IMAGES} images, got {n}")

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
                imgs_flat = imgs5.view(b * n, c, h, w)
                i_tok = clip_encode_image_safe(clip_model, imgs_flat).float()  # [B*5, D]
                i_tok = i_tok.view(b, n, -1)  # [B,5,D]

                # mean multiple pooling
                imgset = mean_multiple_pool(i_tok, POOL_MODE)

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



    # save
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(
        {
            "clip_model_name": CLIP_MODEL_NAME,
            "context_length": int(clip_model.context_length),
            "clip_state": clip_model.state_dict(),
            "pool_mode": POOL_MODE,
        },
        SAVE_PATH
    )

    print(f"Saved: {SAVE_PATH}")

    clip_model.eval()
    with torch.no_grad():
        text_embs, imgset_embs = build_embeddings(test_loader, clip_model, device, USE_FP16)
        metrics = retrieval_metrics_multi(
            text_embs, imgset_embs,
            logit_scale=clip_model.logit_scale,
            device=device,
            text_chunk=TEXT_CHUNK,
            cand_chunk=CAND_CHUNK,
        )
    print(
        f"\n[Epoch {epoch}] Mean-Multiple Retrieval on CLIP {CLIP_MODEL_NAME}:\n"
        f"N={int(metrics['N'])}\n"
        f"R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}\n"
        f"MedianRank={metrics['MedianRank']:.1f}\n"
    )


if __name__ == "__main__":
    main()
