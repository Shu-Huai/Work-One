#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mean Multiple Alignment + Caption Aux Loss (GoodNews / NewsStories style)

主任务（与你现在的检索目标一致）：
- 每条样本：长文章 article + 5 张图
- 图像：CLIP encode_image 得到每张图 embedding，再 mean pooling 得到 image-set embedding
- 文本：CLIP encode_text 得到 article embedding
- 用 CLIP-style 双向 InfoNCE 做 (article <-> imgset) 对齐

新增：Caption 辅助对齐（方案1）
- 你数据里每张图都有 caption（JSON 的 images["0".."4"] 是文字）
- 对 batch 内所有图片摊平到 M=B*5：
    i_flat: [M, D]  (每张图 embedding)
    c_flat: [M, D]  (每张 caption embedding)
  再做 CLIP-style 双向 InfoNCE
- 总损失：loss = loss_ai + LAMBDA_CAP * loss_ic

重要：修复 dtype mismatch
- 这里实现 clip_encode_text_safe / clip_encode_image_safe，matmul 前对齐 dtype
- 只 finetune projection layers（+ logit_scale），其余冻结（你可以按需改）

依赖：
- pip install git+https://github.com/openai/CLIP.git
- 你的 dataset.py 里包含你贴出来的 CaptioningDataset / captioning_collate_fn
"""

import os
import math
import random
from typing import List, Dict, Tuple
from utils import set_seed

import torch
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

# Mean Multiple pooling 方式
POOL_MODE = "mean_normed"  # "mean_normed" or "mean_raw"

# Caption 辅助损失权重（方案1核心超参）
LAMBDA_CAP = 0.6

# 保存
SAVE_NAME = CLIP_MODEL_NAME.replace("/", "").replace("-", "")
SAVE_PATH = f"./ckpt/mean_multiple_caploss_{SAVE_NAME}.pt"

# 评测分块（N^2）
EVAL_EVERY = 1
TEXT_CHUNK = 32
CAND_CHUNK = 512

REQUIRE_NUM_IMAGES = 5
# ============================================================


# --------------------- utils ---------------------


def cosine_with_warmup_lr(step: int, total_steps: int, warmup_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    warmup_steps = min(warmup_steps, total_steps)
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def extend_clip_context_length(model, new_len: int):
    """
    77 -> 256：插值 positional_embedding + 重建 attn_mask，并同步到每个 block
    """
    import torch.nn.functional as F

    if new_len == model.context_length:
        return
    old_len = model.context_length
    if new_len < old_len:
        raise ValueError(f"new_len({new_len}) must be >= old_len({old_len})")

    device = model.positional_embedding.device
    dtype = model.positional_embedding.dtype

    with torch.no_grad():
        old_pe = model.positional_embedding.detach()  # [old_len, width]
        pe = old_pe.T.unsqueeze(0)                    # [1, width, old_len]
        pe_new = F.interpolate(pe, size=new_len, mode="linear", align_corners=False)
        pe_new = pe_new.squeeze(0).T.contiguous()     # [new_len, width]

    model.context_length = new_len
    model.positional_embedding = torch.nn.Parameter(pe_new.to(device=device, dtype=dtype))

    attn_mask = model.build_attention_mask().to(device=device)
    model.attn_mask = attn_mask
    model._buffers["attn_mask"] = attn_mask
    for blk in model.transformer.resblocks:
        blk.attn_mask = attn_mask


def freeze_to_projection_only(model):
    """
    贴一些论文/复现习惯：只 finetune projection layers（+ logit_scale）
    """
    for p in model.parameters():
        p.requires_grad = False

    if getattr(model, "text_projection", None) is not None:
        model.text_projection.requires_grad = True
    if getattr(model, "logit_scale", None) is not None:
        model.logit_scale.requires_grad = True
    if hasattr(model, "visual") and getattr(model.visual, "proj", None) is not None:
        model.visual.proj.requires_grad = True


def cast_trainable_params_to_fp32(model):
    """
    避免 GradScaler 在 fp16 参数上 unscale 报错：把 requires_grad 的参数转 fp32
    """
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()


# --------------------- CLIP safe encoders (dtype align) ---------------------
def clip_encode_text_safe(clip_model, tokens: torch.Tensor) -> torch.Tensor:
    """
    等价于 clip_model.encode_text，但在 x @ text_projection 前对齐 dtype
    """
    x = clip_model.token_embedding(tokens).type(clip_model.dtype)
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)
    x = clip_model.ln_final(x).type(clip_model.dtype)

    eot = tokens.argmax(dim=-1)
    x = x[torch.arange(x.shape[0], device=x.device), eot]  # [B, width]

    proj = clip_model.text_projection
    if proj is not None and proj.dtype != x.dtype:
        proj = proj.to(x.dtype)
    x = x @ proj
    return x


def clip_encode_image_safe(clip_model, images: torch.Tensor) -> torch.Tensor:
    """
    ViT 视觉塔：在 x @ visual.proj 前对齐 dtype
    """
    visual = clip_model.visual
    if visual.__class__.__name__ == "VisionTransformer":
        x = images.type(visual.conv1.weight.dtype)
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        cls = visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype)
        x = torch.cat([cls, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)

        x = visual.ln_post(x[:, 0, :])

        proj = visual.proj
        if proj is not None and proj.dtype != x.dtype:
            proj = proj.to(x.dtype)
        if proj is not None:
            x = x @ proj
        return x

    return clip_model.encode_image(images)


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


def clip_infonce_bidir(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: [N,N]，GT 是同 index
    """
    n = logits.size(0)
    targets = torch.arange(n, device=logits.device)
    loss_a = F.cross_entropy(logits, targets)
    loss_b = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_a + loss_b)


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


@torch.no_grad()
def retrieval_metrics(
    text_embs_cpu: torch.Tensor,    # [N,D]
    imgset_embs_cpu: torch.Tensor,  # [N,D]
    logit_scale: torch.Tensor,
    device: torch.device,
    text_chunk: int,
    cand_chunk: int,
) -> Dict[str, float]:
    """
    text->imgset 检索，GT 为同 index
    """
    N, D = text_embs_cpu.shape
    ranks = torch.empty(N, dtype=torch.long)
    scale = logit_scale.exp().detach().float().to(device)

    img_all = imgset_embs_cpu.to(device)  # [N,D]

    for t0 in tqdm(range(0, N, text_chunk), desc="Scoring (texts)"):
        t1 = min(t0 + text_chunk, N)
        t = text_embs_cpu[t0:t1].to(device)  # [C,D]
        C = t.shape[0]

        scores = torch.empty((C, N), dtype=torch.float32)  # CPU

        for c0 in range(0, N, cand_chunk):
            c1 = min(c0 + cand_chunk, N)
            cand = img_all[c0:c1]  # [M,D]
            sim = (t @ cand.t()) * scale
            scores[:, c0:c1] = sim.detach().cpu()

        gt_idx = torch.arange(t0, t1)
        row = torch.arange(0, C)
        gt_score = scores[row, gt_idx]
        rank = (scores > gt_score.unsqueeze(1)).sum(dim=1) + 1
        ranks[t0:t1] = rank

    return {
        "N": float(N),
        "R@1": float((ranks <= 1).float().mean().item()),
        "R@5": float((ranks <= 5).float().mean().item()),
        "R@10": float((ranks <= 10).float().mean().item()),
        "MedianRank": float(ranks.median().item()),
    }


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
    print(f"Caption aux loss: lambda={LAMBDA_CAP}")

    amp_enabled = (device.type == "cuda" and USE_FP16)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None

    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        clip_model.train()

        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            articles: List[str] = batch["articles"]
            imgs_list: List[torch.Tensor] = batch["images"]       # list of [5,3,H,W]
            caps_list: List[List[str]] = batch["captions"]        # list of list[str], each len=5

            imgs5 = torch.stack(imgs_list, dim=0).to(device)  # [B,5,3,H,W]
            b, n, c, h, w = imgs5.shape
            if n != REQUIRE_NUM_IMAGES:
                raise ValueError(f"Expect {REQUIRE_NUM_IMAGES} images, got {n}")
            if len(caps_list) != b:
                raise ValueError(f"captions batch size mismatch: {len(caps_list)} vs {b}")
            for caps in caps_list:
                if len(caps) != n:
                    raise ValueError(f"Expect {n} captions per sample, got {len(caps)}")

            # tokenize article
            art_tokens = clip.tokenize(
                articles,
                context_length=clip_model.context_length,
                truncate=TEXT_TRUNCATE
            ).to(device)

            # tokenize captions (flatten to B*5)
            caps_flat = [cap.strip() for caps in caps_list for cap in caps]  # len=B*5
            cap_tokens = clip.tokenize(
                caps_flat,
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
                # ===== encode article =====
                t = clip_encode_text_safe(clip_model, art_tokens).float()  # [B,D]
                t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)

                # ===== encode images =====
                imgs_flat = imgs5.view(b * n, c, h, w)
                i_flat = clip_encode_image_safe(clip_model, imgs_flat).float()  # [B*5, D]
                i_flat = i_flat / (i_flat.norm(dim=-1, keepdim=True) + 1e-8)
                i_tok = i_flat.view(b, n, -1)  # [B,5,D]

                # mean multiple pooling -> imgset
                imgset = mean_multiple_pool(i_tok, POOL_MODE)  # [B,D]

                # ===== 主任务：article <-> imgset =====
                logit_scale = clip_model.logit_scale.exp().float()
                logits_ai = logit_scale * (t @ imgset.t())      # [B,B]
                loss_ai = clip_infonce_bidir(logits_ai)

                # ===== 辅助任务：image <-> caption =====
                c_flat = clip_encode_text_safe(clip_model, cap_tokens).float()  # [B*5,D]
                c_flat = c_flat / (c_flat.norm(dim=-1, keepdim=True) + 1e-8)

                logits_ic = logit_scale * (i_flat @ c_flat.t())  # [B*5, B*5]
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
                )

        # 每个 epoch 评测一次（可选）


    # save
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(
        {
            "clip_model_name": CLIP_MODEL_NAME,
            "context_length": int(clip_model.context_length),
            "clip_state": clip_model.state_dict(),
            "pool_mode": POOL_MODE,
            "lambda_cap": float(LAMBDA_CAP),
        },
        SAVE_PATH
    )
    print(f"Saved: {SAVE_PATH}")
    clip_model.eval()
    with torch.no_grad():
        text_embs, imgset_embs = build_embeddings(test_loader, clip_model, device, USE_FP16)
        metrics = retrieval_metrics(
            text_embs, imgset_embs,
            logit_scale=clip_model.logit_scale,
            device=device,
            text_chunk=TEXT_CHUNK,
            cand_chunk=CAND_CHUNK,
        )
    print(
        f"\n[Epoch {epoch}] Retrieval (Text->ImgSet) on CLIP {CLIP_MODEL_NAME}:\n"
        f"N={int(metrics['N'])}\n"
        f"R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}\n"
        f"MedianRank={metrics['MedianRank']:.1f}\n"
    )

if __name__ == "__main__":
    main()
