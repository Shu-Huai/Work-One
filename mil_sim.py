#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIL-SIM baseline reproduction for NewsStories/GoodNews (ECCV'22: NewsStories: Illustrating Articles with Visual Summaries)

核心（按论文 4.2 / Eq.(6) 实现）：
- 把每篇文章切成句子：L={x1,...,x_NL}（默认用 NLTK；无 NLTK 则用简单规则兜底）
- 编码：
  - 每张图 -> yi（CLIP image encoder）
  - 每个句子 -> x_l（CLIP text encoder）
- article-level loss：用 mean-pool 得到 If / Lf，然后做 CLIP-style 双向 InfoNCE
- image-sentence loss：对每张图 yi，计算其与 batch 内每篇文章 L 的 bag 相似度：
      sim(yi, L_k) = max_{l in L_k} <yi, x_l>
  然后对候选文章（k=1..B）做 softmax cross-entropy（正样本是同 index 的文章）
- 总损失：L = L_article + LAMBDA * L_imgsent

依赖：
  pip install ftfy regex tqdm
  pip install git+https://github.com/openai/CLIP.git
  (可选) pip install nltk && python -c "import nltk; nltk.download('punkt')"

备注：
- 贴你的其他脚本：冻结 CLIP backbone，只训练 projection layers + logit_scale（论文训练设置）
- 只做 train/test，不管 5 split / val（按你的要求）
"""

import os
import math
import random
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import set_seed

import clip

# dataset import（跟你仓库一致）
from dataset import CaptioningDataset, captioning_collate_fn


# ===================== 你只需要改这些参数 =====================
# 数据
REQUIRE_NUM_IMAGES = 4

TRAIN_JSON_PATH = f"data/captioning_dataset_{REQUIRE_NUM_IMAGES}imgs_train_60.json"
TEST_JSON_PATH  = f"data/captioning_dataset_{REQUIRE_NUM_IMAGES}imgs_test_40.json"
IMAGE_ROOT = "data/resized"

# CLIP
CLIP_MODEL_NAME = "ViT-L/14@336px"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = True

# 论文里做了长度 ablation；你想贴论文可用 256
TEXT_CONTEXT_LENGTH = 256
TEXT_TRUNCATE = True

# 句子切分 & 句子数控制（用参数决定）
SENT_SPLITTER = "nltk"     # "nltk" | "simple"
MAX_SENTENCES = 64         # 0 表示不截断
SENT_SELECT = "random"      # "first" | "random"
MIN_SENT_CHARS = 32         # 过短句子直接丢弃（兜底，避免空 token）

# 训练
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

# MIL-SIM trade-off（论文说“reduce the weight”，默认 0.1）
LAMBDA = 0.1

# 评测分块（N^2）
EVAL_EVERY = 1
EVAL_MIN_EPOCH = 20
TEXT_CHUNK = 32
CAND_CHUNK = 512

# 如果你的数据固定 5 张图就保持 5；否则改成 0 并自己改 dataset/collate

SAVE_NAME = CLIP_MODEL_NAME.replace("/", "").replace("-", "")
SAVE_PATH = f"./ckpt/mil_sim_{SAVE_NAME}.pt"
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
    """77 -> 256：插值 positional_embedding + 重建 attn_mask，并同步到每个 block"""
    import torch.nn.functional as F

    if new_len == model.context_length:
        return
    old_len = model.context_length
    if new_len < old_len:
        raise ValueError(f"new_len({new_len}) must be >= old_len({old_len})")

    device = model.positional_embedding.device
    dtype = model.positional_embedding.dtype

    with torch.no_grad():
        old_pe = model.positional_embedding.detach()          # [old_len, width]
        pe = old_pe.T.unsqueeze(0)                            # [1, width, old_len]
        pe_new = F.interpolate(pe, size=new_len, mode="linear", align_corners=False)
        pe_new = pe_new.squeeze(0).T.contiguous()             # [new_len, width]

    model.context_length = new_len
    model.positional_embedding = torch.nn.Parameter(pe_new.to(device=device, dtype=dtype))

    attn_mask = model.build_attention_mask().to(device=device)
    model.attn_mask = attn_mask
    model._buffers["attn_mask"] = attn_mask
    for blk in model.transformer.resblocks:
        blk.attn_mask = attn_mask


def freeze_to_projection_only(model):
    """只 finetune projection layers（+ logit_scale）"""
    for p in model.parameters():
        p.requires_grad = False

    if getattr(model, "text_projection", None) is not None:
        model.text_projection.requires_grad = True
    if getattr(model, "logit_scale", None) is not None:
        model.logit_scale.requires_grad = True
    if hasattr(model, "visual") and getattr(model.visual, "proj", None) is not None:
        model.visual.proj.requires_grad = True


def cast_trainable_params_to_fp32(model):
    """避免 GradScaler 在 fp16 参数上 unscale 报错：把 requires_grad 的参数转 fp32"""
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()


# --------------------- CLIP safe encoders (dtype align) ---------------------
def clip_encode_text_safe(clip_model, tokens: torch.Tensor) -> torch.Tensor:
    """等价于 clip_model.encode_text，但在 x @ text_projection 前对齐 dtype"""
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
    """ViT 视觉塔：在 x @ visual.proj 前对齐 dtype"""
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


# --------------------- sentence split ---------------------
def _split_sentences_nltk(text: str) -> List[str]:
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
        # 有些环境没下 punkt，会抛 LookupError
        try:
            sents = sent_tokenize(text)
        except LookupError:
            # 尽量自动下载（失败也不影响：回退 simple）
            try:
                nltk.download("punkt", quiet=True)
                sents = sent_tokenize(text)
            except Exception:
                return _split_sentences_simple(text)
        return sents
    except Exception:
        return _split_sentences_simple(text)


def _split_sentences_simple(text: str) -> List[str]:
    # 一个比较稳的兜底：按 .!? 换行切分
    import re
    text = text.replace("\r", "\n")
    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]

# 1) 配置注释也改一下
# SENT_SELECT = "first"  # "first" | "random" | "uniform"

def _uniform_pick(sents: List[str], k: int) -> List[str]:
    """
    从 n 个句子里均匀取 k 个（确定性、保持顺序）。
    仅在 n > k 时调用。
    """
    n = len(sents)
    if k <= 0 or k >= n:
        return sents
    if k == 1:
        return [sents[0]]
    step = (n - 1) / (k - 1)  # >= 1 when n >= k
    idxs = [int(math.floor(i * step)) for i in range(k)]  # 0 ... n-1，且严格递增
    return [sents[i] for i in idxs]


def split_sentences(text: str) -> List[str]:
    if SENT_SPLITTER == "nltk":
        sents = _split_sentences_nltk(text)
    else:
        sents = _split_sentences_simple(text)

    # 过滤太短
    sents = [s.strip() for s in sents if s and len(s.strip()) >= MIN_SENT_CHARS]
    if len(sents) == 0:
        # 极端兜底：把全文当成 1 句
        sents = [text.strip()[:1000] if text else "empty"]

    if MAX_SENTENCES and MAX_SENTENCES > 0 and len(sents) > MAX_SENTENCES:
        if SENT_SELECT == "random":
            # 如果你希望 random 也保持原顺序：采样 index 后排序
            idxs = sorted(random.sample(range(len(sents)), MAX_SENTENCES))
            sents = [sents[i] for i in idxs]
        elif SENT_SELECT == "uniform":
            sents = _uniform_pick(sents, MAX_SENTENCES)
        else:  # first
            sents = sents[:MAX_SENTENCES]


def mean_pool_normed(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    x: [..., D] (已 L2 norm)
    return: mean -> L2 norm
    """
    out = x.mean(dim=dim)
    out = out / (out.norm(dim=-1, keepdim=True) + 1e-8)
    return out


# --------------------- eval embedding & metrics ---------------------
@torch.no_grad()
def build_embeddings_milsim(
    dataloader: DataLoader,
    clip_model,
    device: torch.device,
    use_fp16: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    输出 CPU embeddings：
      text_embs  : [N, D]   (Lf: mean over sentences)
      imgset_embs: [N, D]   (If: mean over images)
    """
    clip_model.eval()

    amp_enabled = (device.type == "cuda" and use_fp16)
    autocast_ctx = torch.amp.autocast("cuda", enabled=amp_enabled) if device.type == "cuda" \
                   else torch.amp.autocast("cpu", enabled=False)

    all_t, all_i = [], []
    for batch in tqdm(dataloader, desc="Build embeddings (MIL-SIM)"):
        articles: List[str] = batch["articles"]
        imgs_list: List[torch.Tensor] = batch["images"]  # list of [N,3,H,W] (你的数据是 5)

        imgs5 = torch.stack(imgs_list, dim=0).to(device)  # [B,N,3,H,W]
        b, n, c, h, w = imgs5.shape
        if REQUIRE_NUM_IMAGES and REQUIRE_NUM_IMAGES > 0 and n != REQUIRE_NUM_IMAGES:
            raise ValueError(f"Expect {REQUIRE_NUM_IMAGES} images, got {n}")

        # sentences
        sent_lists = [split_sentences(a) for a in articles]
        sent_offsets = [0]
        flat_sents: List[str] = []
        for sents in sent_lists:
            flat_sents.extend(sents)
            sent_offsets.append(len(flat_sents))

        with autocast_ctx:
            # encode all sentences
            sent_tokens = clip.tokenize(
                flat_sents,
                context_length=clip_model.context_length,
                truncate=TEXT_TRUNCATE
            ).to(device)
            sent_emb = clip_encode_text_safe(clip_model, sent_tokens).float()
            sent_emb = sent_emb / (sent_emb.norm(dim=-1, keepdim=True) + 1e-8)  # [S,D]

            # article-level Lf
            Lf_list = []
            for i in range(b):
                s0, s1 = sent_offsets[i], sent_offsets[i + 1]
                Lf_list.append(mean_pool_normed(sent_emb[s0:s1], dim=0))
            Lf = torch.stack(Lf_list, dim=0)  # [B,D]

            # images yi
            imgs_flat = imgs5.view(b * n, c, h, w)
            yi = clip_encode_image_safe(clip_model, imgs_flat).float()
            yi = yi / (yi.norm(dim=-1, keepdim=True) + 1e-8)
            yi = yi.view(b, n, -1)

            # image-set If
            If = mean_pool_normed(yi, dim=1)  # [B,D]

        all_t.append(Lf.cpu())
        all_i.append(If.cpu())

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
    """text->imgset 检索，GT 为同 index"""
    N, D = text_embs_cpu.shape
    ranks = torch.empty(N, dtype=torch.long)
    scale = logit_scale.exp().detach().float().to(device)

    img_all = imgset_embs_cpu.to(device)

    for t0 in tqdm(range(0, N, text_chunk), desc="Scoring (texts)"):
        t1 = min(t0 + text_chunk, N)
        t = text_embs_cpu[t0:t1].to(device)  # [C,D]
        C = t.shape[0]

        scores = torch.empty((C, N), dtype=torch.float32)  # CPU

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


# --------------------- train step: MIL-SIM ---------------------
def milsim_losses(
    clip_model,
    img_emb: torch.Tensor,          # [B, NI, D]  yi (normed)
    sent_emb: torch.Tensor,         # [S_total, D] sentence embeddings (normed)
    sent_offsets: List[int],        # len=B+1, cumulative
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回：(loss_article, loss_imgsent)
    """
    device = img_emb.device
    b, ni, d = img_emb.shape

    # 1) article-level: If / Lf mean pooling
    If = mean_pool_normed(img_emb, dim=1)  # [B,D]
    Lf_list = []
    for i in range(b):
        s0, s1 = sent_offsets[i], sent_offsets[i + 1]
        Lf_list.append(mean_pool_normed(sent_emb[s0:s1], dim=0))
    Lf = torch.stack(Lf_list, dim=0)  # [B,D]

    logit_scale = clip_model.logit_scale.exp().float()

    logits = logit_scale * (Lf @ If.t())  # [B,B]
    targets = torch.arange(b, device=device)
    loss_t = F.cross_entropy(logits, targets)
    loss_i = F.cross_entropy(logits.t(), targets)
    loss_article = 0.5 * (loss_t + loss_i)

    # 2) image-sentence: each image -> bag of sentences (max over sentences per article)
    # sims: [B*NI, S_total]
    img_flat = img_emb.reshape(b * ni, d)  # [Q,D]
    sims = img_flat @ sent_emb.t()         # [Q,S]

    # bag_logits[q, k] = max_{l in article k} sims[q, s0:s1]
    Q = sims.shape[0]
    bag_logits = torch.empty((Q, b), device=device, dtype=sims.dtype)
    for k in range(b):
        s0, s1 = sent_offsets[k], sent_offsets[k + 1]
        bag_logits[:, k] = sims[:, s0:s1].max(dim=1).values

    bag_logits = bag_logits * logit_scale  # temperature
    labels = torch.arange(b, device=device).repeat_interleave(ni)  # [Q]
    loss_imgsent = F.cross_entropy(bag_logits, labels)

    return loss_article, loss_imgsent


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

    # 4) optimizer
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
        shuffle=False,  # 必须 false，保证 index 对齐 GT
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY if device.type == "cuda" else False,
        collate_fn=captioning_collate_fn,
        drop_last=False,
    )

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = min(WARMUP_STEPS, max(0, total_steps // 10))
    print(f"Train size={len(train_ds)}, Test size={len(test_ds)}")
    print(f"Total steps={total_steps}, Warmup steps={warmup_steps}")
    print(f"CLIP={CLIP_MODEL_NAME}  ctx={clip_model.context_length}  lambda={LAMBDA}")
    print(f"Sentence: splitter={SENT_SPLITTER}, max={MAX_SENTENCES}, select={SENT_SELECT}, min_chars={MIN_SENT_CHARS}")

    amp_enabled = (device.type == "cuda" and USE_FP16)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None

    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        clip_model.train()

        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            articles: List[str] = batch["articles"]
            imgs_list: List[torch.Tensor] = batch["images"]  # list of [N,3,H,W]

            imgs = torch.stack(imgs_list, dim=0).to(device)  # [B,N,3,H,W]
            b, n, c, h, w = imgs.shape
            if REQUIRE_NUM_IMAGES and REQUIRE_NUM_IMAGES > 0 and n != REQUIRE_NUM_IMAGES:
                raise ValueError(f"Expect {REQUIRE_NUM_IMAGES} images, got {n}")

            # sentences
            sent_lists = [split_sentences(a) for a in articles]
            sent_offsets = [0]
            flat_sents: List[str] = []
            for sents in sent_lists:
                flat_sents.extend(sents)
                sent_offsets.append(len(flat_sents))

            lr_scale = cosine_with_warmup_lr(global_step, total_steps, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = LR * lr_scale

            optimizer.zero_grad(set_to_none=True)

            autocast_ctx = torch.amp.autocast("cuda", enabled=amp_enabled) if device.type == "cuda" \
                           else torch.amp.autocast("cpu", enabled=False)

            with autocast_ctx:
                # encode sentences
                sent_tokens = clip.tokenize(
                    flat_sents,
                    context_length=clip_model.context_length,
                    truncate=TEXT_TRUNCATE
                ).to(device)
                sent_emb = clip_encode_text_safe(clip_model, sent_tokens).float()
                sent_emb = sent_emb / (sent_emb.norm(dim=-1, keepdim=True) + 1e-8)  # [S,D]

                # encode images
                imgs_flat = imgs.view(b * n, c, h, w)
                img_emb = clip_encode_image_safe(clip_model, imgs_flat).float()
                img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-8)
                img_emb = img_emb.view(b, n, -1)

                loss_article, loss_imgsent = milsim_losses(
                    clip_model=clip_model,
                    img_emb=img_emb,
                    sent_emb=sent_emb,
                    sent_offsets=sent_offsets,
                )
                loss = loss_article + float(LAMBDA) * loss_imgsent

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
                    la=float(loss_article.item()),
                    ls=float(loss_imgsent.item()),
                    lr=float(optimizer.param_groups[0]["lr"]),
                )

        if epoch >= EVAL_MIN_EPOCH and epoch % EVAL_EVERY == 0:
            clip_model.eval()
            with torch.no_grad():
                text_embs, imgset_embs = build_embeddings_milsim(test_loader, clip_model, device, USE_FP16)
                metrics = retrieval_metrics(
                    text_embs, imgset_embs,
                    logit_scale=clip_model.logit_scale,
                    device=device,
                    text_chunk=TEXT_CHUNK,
                    cand_chunk=CAND_CHUNK,
                )
            print(
                f"\n[Epoch {epoch}] MIL-SIM Retrieval on CLIP {CLIP_MODEL_NAME}:\n"
                f"N={int(metrics['N'])}\n"
                f"R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}\n"
                f"MedianRank={metrics['MedianRank']:.1f}\n"
            )
    # save
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(
        {
            "clip_model_name": CLIP_MODEL_NAME,
            "context_length": int(clip_model.context_length),
            "clip_state": clip_model.state_dict(),
            "lambda": float(LAMBDA),
            "max_sentences": int(MAX_SENTENCES),
            "sent_select": str(SENT_SELECT),
            "sent_splitter": str(SENT_SPLITTER),
        },
        SAVE_PATH
    )
    print(f"Saved: {SAVE_PATH}")



if __name__ == "__main__":
    main()
