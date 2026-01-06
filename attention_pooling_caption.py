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
from utils import set_seed

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
        old_pe = model.positional_embedding.detach()
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
    只 finetune projection layers（+ logit_scale）
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


# --------------------- CLIP safe encoders (dtype align) ---------------------
def clip_encode_text_safe(clip_model, tokens: torch.Tensor) -> torch.Tensor:
    x = clip_model.token_embedding(tokens).type(clip_model.dtype)
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)
    x = clip_model.ln_final(x).type(clip_model.dtype)

    eot = tokens.argmax(dim=-1)
    x = x[torch.arange(x.shape[0], device=x.device), eot]

    proj = clip_model.text_projection
    if proj is not None and proj.dtype != x.dtype:
        proj = proj.to(x.dtype)
    x = x @ proj
    return x


def clip_encode_image_safe(clip_model, images: torch.Tensor) -> torch.Tensor:
    """
    ViT：手写 forward（dtype 对齐）
    RN：走 clip_model.encode_image（最小改动）
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


@torch.no_grad()
def retrieval_metrics(
    text_embs_cpu: torch.Tensor,
    imgset_embs_cpu: torch.Tensor,
    logit_scale: torch.Tensor,
    device: torch.device,
    text_chunk: int,
    cand_chunk: int,
) -> Dict[str, float]:
    N, D = text_embs_cpu.shape
    ranks = torch.empty(N, dtype=torch.long)
    scale = logit_scale.exp().detach().float().to(device)

    img_all = imgset_embs_cpu.to(device)

    for t0 in tqdm(range(0, N, text_chunk), desc="Scoring (texts)"):
        t1 = min(t0 + text_chunk, N)
        t = text_embs_cpu[t0:t1].to(device)
        C = t.shape[0]

        scores = torch.empty((C, N), dtype=torch.float32)

        for c0 in range(0, N, cand_chunk):
            c1 = min(c0 + cand_chunk, N)
            cand = img_all[c0:c1]
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
        metrics = retrieval_metrics(
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
