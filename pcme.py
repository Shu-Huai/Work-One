#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCME baseline reproduction for NewsStories/GoodNews multi-image long-article alignment.

论文要点（在你的仓库里对应的实现约束）：
- 数据：每条样本 = (article text, 5 images)
- 训练：跟 Single Image 一样，每条样本随机取 1 张图来训练（更贴近论文里的 "Single" 设置）
- PCME：把 image/text 表示成高斯分布；用 K 次采样做 soft-match 概率并做 BCE 对比学习
- 测试：相似度用“最高分的采样对”（max over K^2），再对 5 张图取均值作为 image-set 分数

注意：
- 代码默认依赖你已有的 dataset.py（脚本里和你给的文件一致：from dataset import CaptioningDataset, captioning_collate_fn）
  如果你文件名叫 GoodNewsDataset.py，请把它改名为 dataset.py 或者改下面 import。
"""
import math
import random
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import set_seed

import clip
from dataset import CaptioningDataset, captioning_collate_fn


# ===================== 你只需要改这些参数 =====================
# 数据
TRAIN_JSON_PATH = "data/captioning_dataset_5imgs_train_60.json"
TEST_JSON_PATH  = "data/captioning_dataset_5imgs_test_40.json"
IMAGE_ROOT = "data/resized"

# CLIP
CLIP_MODEL_NAME = "ViT-L/14@336px"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = True

# 论文里会扩到 256（需要改 positional embedding + attn_mask + 每个block的mask）
TEXT_CONTEXT_LENGTH = 256   # 77 或 256
TEXT_TRUNCATE = True
KL_WEIGHT = 1e-5
# 训练
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 4
PIN_MEMORY = True

EPOCHS = 20
LR = 5e-3
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 20000   # 数据小会自动缩短
GRAD_CLIP_NORM = 1.0
PRINT_EVERY = 20

# PCME 采样数：训练 / 测试可分开（测试 K 越大越慢）
TRAIN_K = 8
EVAL_K  = 8

# log-variance 的范围（防数值爆炸）
LOGVAR_MIN = -10.0
LOGVAR_MAX = 2.0

# 保存
SAVE_NAME = CLIP_MODEL_NAME.replace("/","").replace("-","")
SAVE_PATH = f"./ckpt/pcme_clip_{SAVE_NAME}.pt"

# 测试评测分块
TEXT_CHUNK = 32          # text chunk（太慢就调大；显存不够就调小）
CAND_CHUNK = 512         # candidate chunk（显存不够就调小）
REQUIRE_NUM_IMAGES = 5
# ============================================================




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
    安全扩展 openai/CLIP 的 context length（77 -> 256）：
    - positional_embedding 线性插值
    - model.attn_mask / buffer
    - 每个 transformer block 的 attn_mask
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
        old_pe = model.positional_embedding.detach()      # [old_len, width]
        pe = old_pe.T.unsqueeze(0)                        # [1, width, old_len]
        pe_new = F.interpolate(pe, size=new_len, mode="linear", align_corners=False)
        pe_new = pe_new.squeeze(0).T.contiguous()         # [new_len, width]

    model.context_length = new_len
    model.positional_embedding = torch.nn.Parameter(pe_new.to(device=device, dtype=dtype))

    attn_mask = model.build_attention_mask().to(device=device)
    model.attn_mask = attn_mask
    model._buffers["attn_mask"] = attn_mask

    for blk in model.transformer.resblocks:
        blk.attn_mask = attn_mask


def freeze_to_projection_only(model):
    """
    只训练 projection layers（贴你 single/mil_nce 的风格）
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
    把 requires_grad=True 的参数转 fp32（避免 GradScaler 的 unscale FP16 梯度问题）
    """
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()


def clip_encode_text_safe(clip_model, tokens: torch.Tensor) -> torch.Tensor:
    """
    等价于 clip_model.encode_text，但会在 projection matmul 前做 dtype 对齐，
    解决 x(fp16) @ text_projection(fp32) 报错。
    """
    x = clip_model.token_embedding(tokens).type(clip_model.dtype)   # [B, T, width]
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    # take features from the eot embedding
    eot = tokens.argmax(dim=-1)
    x = x[torch.arange(x.shape[0], device=x.device), eot]           # [B, width]

    proj = clip_model.text_projection           # Parameter
    x = x.to(proj.dtype)                        # cast x, NOT proj
    x = x @ proj
    return x


def clip_encode_image_safe(clip_model, images: torch.Tensor) -> torch.Tensor:
    """
    ViT 视觉塔：在 visual.proj matmul 前做 dtype 对齐，解决 fp16 @ fp32 报错。
    """
    visual = clip_model.visual
    if visual.__class__.__name__ == "VisionTransformer":
        x = images.type(visual.conv1.weight.dtype)
        x = visual.conv1(x)                         # [B, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)   # [B, width, grid**2]
        x = x.permute(0, 2, 1)                      # [B, grid**2, width]

        cls = visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype)
        x = torch.cat([cls, x], dim=1)              # [B, 1+grid**2, width]
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)

        x = x.permute(1, 0, 2)                      # NLD -> LND
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)                      # LND -> NLD

        x = visual.ln_post(x[:, 0, :])

        proj = visual.proj                          # Parameter
        x = x.to(proj.dtype)                        # cast x, NOT proj
        if proj is not None:
            x = x @ proj
        return x

    # 如果你换 RN50/RN101 等：一般不会触发该问题；若你也把 RN 的 proj 转 fp32，再来我给你补 safe forward
    return clip_model.encode_image(images)


class PCMEWrapper(nn.Module):
    """
    PCME 头：
    - mu：用 CLIP 的 encode 输出（再 L2 norm）
    - logvar：各自一个 Linear(D->D)
    - alpha/beta：match probability 的标量参数（alpha 用 softplus 保证 >0）
    """
    def __init__(self, clip_model, embed_dim: int):
        super().__init__()
        self.clip = clip_model

        self.txt_logvar = nn.Linear(embed_dim, embed_dim)
        self.img_logvar = nn.Linear(embed_dim, embed_dim)

        # 初始化：先让方差小一点
        nn.init.zeros_(self.txt_logvar.weight)
        nn.init.zeros_(self.img_logvar.weight)
        nn.init.constant_(self.txt_logvar.bias, -4.0)
        nn.init.constant_(self.img_logvar.bias, -4.0)

        self.alpha_raw = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def alpha(self) -> torch.Tensor:
        return F.softplus(self.alpha_raw) + 1e-6

    def encode_text_mu_logvar(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = clip_encode_text_safe(self.clip, tokens).float()
        mu = mu / (mu.norm(dim=-1, keepdim=True) + 1e-8)
        logvar = self.txt_logvar(mu)
        logvar = torch.clamp(logvar, LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar

    def encode_image_mu_logvar(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = clip_encode_image_safe(self.clip, images).float()
        mu = mu / (mu.norm(dim=-1, keepdim=True) + 1e-8)
        logvar = self.img_logvar(mu)
        logvar = torch.clamp(logvar, LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor, k: int) -> torch.Tensor:
        """
        mu/logvar: [B,D] -> samples: [B,k,D] (每个 sample 再 L2 norm)
        """
        b, d = mu.shape
        std = torch.exp(0.5 * logvar)
        eps = torch.randn((b, k, d), device=mu.device, dtype=mu.dtype)
        z = mu.unsqueeze(1) + eps * std.unsqueeze(1)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)

        return z

    def pcme_prob_matrix_avg(self, zt: torch.Tensor, zi: torch.Tensor) -> torch.Tensor:
        """
        zt: [B,K,D] (normalized)
        zi: [B,K,D] (normalized)
        return p_avg: [B,B] where p_ij = mean_{k,k'} sigmoid(-alpha*dist2 + beta)
        dist2 = 2 - 2*cos
        """
        b, k, d = zt.shape
        zt_flat = zt.reshape(b * k, d)
        zi_flat = zi.reshape(b * k, d)
        dot = zt_flat @ zi_flat.t()                      # [bk, bk]
        zt2 = (zt_flat ** 2).sum(dim=-1, keepdim=True)   # [bk, 1]
        zi2 = (zi_flat ** 2).sum(dim=-1, keepdim=True).t()  # [1, bk]
        dist2 = (zt2 + zi2 - 2 * dot)                    # [bk, bk]
        dist2 = dist2.view(b, k, b, k).permute(0, 2, 1, 3)  # [B,B,K,K]
        logits = -self.alpha() * dist2 + self.beta
        prob = torch.sigmoid(logits)
        return prob.mean(dim=(-1, -2))

    def pcme_max_sim_text_to_imgset(
        self,
        text_samples: torch.Tensor,    # [C,K,D]
        imgset_samples: torch.Tensor,  # [M,5,K,D]
    ) -> torch.Tensor:
        """
        评测用：max over K^2（最高分采样对），再对 5 张图取均值
        输出 [C,M] 的 bag score（概率）
        """
        c, k, d = text_samples.shape
        m, five, k2, d2 = imgset_samples.shape
        assert five == REQUIRE_NUM_IMAGES and k2 == k and d2 == d

        t_flat = text_samples.reshape(c * k, d)              # [C*K, D]
        i_flat = imgset_samples.reshape(m * five * k, d)     # [M*5*K, D]

        sim = t_flat @ i_flat.t()                            # [C*K, M*5*K]
        sim = sim.view(c, k, m, five, k)                     # [C,K,M,5,K]

        # sim: [C,K,M,5,K]
        dist2 = 2.0 - 2.0 * sim                              # [C,K,M,5,K]
        logits = -self.alpha() * dist2 + self.beta           # [C,K,M,5,K]
        prob = torch.sigmoid(logits)                         # [C,K,M,5,K]

        # 平均掉 K^2 -> [C,M,5]，再对 5 张图均值 -> [C,M]
        prob_5 = prob.mean(dim=(1, 4))                       # mean over K and K'
        return prob_5.mean(dim=-1)                           # mean over 5 images


def build_optimizer(pcme: PCMEWrapper):
    params = [p for p in pcme.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)


@torch.no_grad()
def build_eval_samples(
    dataloader: DataLoader,
    pcme: PCMEWrapper,
    device: torch.device,
    eval_k: int,
    text_context_length: int,
    text_truncate: bool,
    use_fp16: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    输出（都在 CPU）：
      text_samples  : [N,K,D]
      imgset_samples: [N,5,K,D]
    """
    pcme.eval()

    amp_enabled = (device.type == "cuda" and use_fp16)
    if device.type == "cuda":
        autocast_ctx = torch.amp.autocast("cuda", enabled=amp_enabled)
    else:
        autocast_ctx = torch.amp.autocast("cpu", enabled=False)

    all_t, all_i = [], []

    for batch in tqdm(dataloader, desc="Build eval samples"):
        articles: List[str] = batch["articles"]
        imgs_list: List[torch.Tensor] = batch["images"]  # list of [5,3,H,W]

        for t in imgs_list:
            if t.ndim != 4 or t.shape[0] != REQUIRE_NUM_IMAGES:
                raise ValueError(f"Expect {REQUIRE_NUM_IMAGES} imgs per sample, got {tuple(t.shape)}")

        tokens = clip.tokenize(
            articles,
            context_length=text_context_length,
            truncate=text_truncate
        ).to(device)

        imgs = torch.stack(imgs_list, dim=0)  # [B,5,3,H,W]
        bsz, five, c, h, w = imgs.shape
        imgs = imgs.view(bsz * five, c, h, w).to(device)

        with autocast_ctx:
            t_mu, t_lv = pcme.encode_text_mu_logvar(tokens)        # [B,D]
            i_mu, i_lv = pcme.encode_image_mu_logvar(imgs)         # [B*5,D]
            t_s = pcme.sample(t_mu, t_lv, eval_k)                  # [B,K,D]
            i_s = pcme.sample(i_mu, i_lv, eval_k)                  # [B*5,K,D]

        i_s = i_s.view(bsz, five, eval_k, -1)                      # [B,5,K,D]
        all_t.append(t_s.cpu())
        all_i.append(i_s.cpu())

    return torch.cat(all_t, dim=0), torch.cat(all_i, dim=0)


@torch.no_grad()
def retrieval_metrics_pcme(
    text_samples_cpu: torch.Tensor,    # [N,K,D] on CPU
    imgset_samples_cpu: torch.Tensor,  # [N,5,K,D] on CPU
    pcme: PCMEWrapper,
    device: torch.device,
    text_chunk: int,
    cand_chunk: int,
) -> Dict[str, float]:
    """
    GT：第 i 个 text 对应第 i 个 image-set（test_loader 必须 shuffle=False）
    """
    pcme.eval()
    N, K, D = text_samples_cpu.shape
    assert imgset_samples_cpu.shape[:3] == (N, REQUIRE_NUM_IMAGES, K)

    ranks = torch.empty(N, dtype=torch.long)

    for t0 in tqdm(range(0, N, text_chunk), desc="Scoring (texts)"):
        t1 = min(t0 + text_chunk, N)
        C = t1 - t0

        scores = torch.empty((C, N), dtype=torch.float32)  # CPU
        t_s = text_samples_cpu[t0:t1].to(device)           # [C,K,D]

        for c0 in range(0, N, cand_chunk):
            c1 = min(c0 + cand_chunk, N)
            i_s = imgset_samples_cpu[c0:c1].to(device)     # [M,5,K,D]
            bag_score = pcme.pcme_max_sim_text_to_imgset(t_s, i_s)  # [C,M]
            scores[:, c0:c1] = bag_score.detach().cpu()

        gt_idx = torch.arange(t0, t1)
        row = torch.arange(0, C)
        gt_score = scores[row, gt_idx]                     # [C]
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

    # 2) extend context length
    if TEXT_CONTEXT_LENGTH != clip_model.context_length:
        extend_clip_context_length(clip_model, TEXT_CONTEXT_LENGTH)

    # 3) freeze CLIP backbone (projection-only) + cast trainable to fp32
    freeze_to_projection_only(clip_model)
    cast_trainable_params_to_fp32(clip_model)

    # 4) infer embed dim（不要 encode_text(dummy)，否则又会触发 dtype mismatch）
    with torch.no_grad():
        d = int(clip_model.text_projection.shape[1])  # [width, embed_dim]

    pcme = PCMEWrapper(clip_model, embed_dim=d).to(device)

    # PCME 头参数要训练
    for p in pcme.txt_logvar.parameters():
        p.requires_grad = True
    for p in pcme.img_logvar.parameters():
        p.requires_grad = True
    pcme.alpha_raw.requires_grad = True
    pcme.beta.requires_grad = True

    # 只对 requires_grad 的参数转 fp32（含 PCME 头）
    for p in pcme.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    optimizer = build_optimizer(pcme)

    # 5) data
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
    print(f"CLIP ctx={clip_model.context_length}, embed_dim={d}")
    print(f"PCME TRAIN_K={TRAIN_K}, EVAL_K={EVAL_K}")

    amp_enabled = (device.type == "cuda" and USE_FP16)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None

    global_step = 0
    pcme.train()

    for epoch in range(1, EPOCHS + 1):
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            articles: List[str] = batch["articles"]
            imgs_list: List[torch.Tensor] = batch["images"]  # list of [5,3,H,W]

            # Single-like: 每条随机取 1 张图
            picked = []
            for imgs in imgs_list:
                if imgs.ndim != 4 or imgs.shape[0] != REQUIRE_NUM_IMAGES:
                    raise ValueError(f"Expect {REQUIRE_NUM_IMAGES} imgs, got {tuple(imgs.shape)}")
                j = random.randint(0, REQUIRE_NUM_IMAGES - 1)
                picked.append(imgs[j])
            images = torch.stack(picked, dim=0).to(device)  # [B,3,H,W]
            bsz = images.size(0)

            tokens = clip.tokenize(
                articles,
                context_length=clip_model.context_length,
                truncate=TEXT_TRUNCATE
            ).to(device)

            # lr schedule
            lr_scale = cosine_with_warmup_lr(global_step, total_steps, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = LR * lr_scale

            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda":
                autocast_ctx = torch.amp.autocast("cuda", enabled=amp_enabled)
            else:
                autocast_ctx = torch.amp.autocast("cpu", enabled=False)

            with autocast_ctx:
                t_mu, t_lv = pcme.encode_text_mu_logvar(tokens)    # [B,D]
                i_mu, i_lv = pcme.encode_image_mu_logvar(images)   # [B,D]

                t_s = pcme.sample(t_mu, t_lv, TRAIN_K)             # [B,K,D]
                i_s = pcme.sample(i_mu, i_lv, TRAIN_K)             # [B,K,D]

                p_mat = pcme.pcme_prob_matrix_avg(t_s, i_s)        # [B,B]

                # balanced BCE：pos=diag, neg=offdiag
                eps = 1e-8
                pos = torch.diagonal(p_mat)                        # [B]
                pos_loss = -torch.log(pos.clamp(min=eps)).mean()

                mask = ~torch.eye(bsz, device=p_mat.device, dtype=torch.bool)
                neg = p_mat[mask]                                  # [B*(B-1)]
                neg_loss = -torch.log((1.0 - neg).clamp(min=eps)).mean()

                loss = 0.5 * (pos_loss + neg_loss)
                # ---- KL regularization (to N(0,I)) to prevent variance collapse ----
                # kl(x) = 0.5 * sum( mu^2 + exp(logvar) - 1 - logvar )
                D = t_mu.shape[-1]
                kl_t = 0.5 * (t_mu.pow(2) + t_lv.exp() - 1.0 - t_lv).sum(dim=-1).mean() / D
                kl_i = 0.5 * (i_mu.pow(2) + i_lv.exp() - 1.0 - i_lv).sum(dim=-1).mean() / D
                loss = loss + KL_WEIGHT * (kl_t + kl_i)
            if device.type == "cuda" and scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in pcme.parameters() if p.requires_grad],
                        GRAD_CLIP_NORM
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in pcme.parameters() if p.requires_grad],
                        GRAD_CLIP_NORM
                    )
                optimizer.step()

            global_step += 1
            if global_step % PRINT_EVERY == 0:
                pbar.set_postfix(
                    loss=float(loss.item()),
                    kl=float((kl_t + kl_i).detach().cpu().item()),
                    alpha=float(pcme.alpha().detach().cpu().item()),
                    beta=float(pcme.beta.detach().cpu().item()),
                    lr=float(optimizer.param_groups[0]["lr"]),
                )

    # save
    ckpt = {
        "clip_model_name": CLIP_MODEL_NAME,
        "context_length": int(clip_model.context_length),
        "embed_dim": int(d),
        "pcme_state": pcme.state_dict(),
    }
    torch.save(ckpt, SAVE_PATH)

    print(f"Saved: {SAVE_PATH}")

    pcme.eval()
    with torch.no_grad():
        t_s_cpu, i_s_cpu = build_eval_samples(
            test_loader,
            pcme,
            device,
            eval_k=EVAL_K,
            text_context_length=clip_model.context_length,
            text_truncate=TEXT_TRUNCATE,
            use_fp16=USE_FP16,
        )
        metrics = retrieval_metrics_pcme(
            t_s_cpu,
            i_s_cpu,
            pcme,
            device,
            text_chunk=TEXT_CHUNK,
            cand_chunk=CAND_CHUNK,
        )
    print(f"\n==== Test Results (PCME trained, Retrieval) using CLIP {CLIP_MODEL_NAME} ====")
    print(f"N={int(metrics['N'])}")
    print(f"R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}")
    print(f"MedianRank={int(metrics['MedianRank'])}")


if __name__ == "__main__":
    main()