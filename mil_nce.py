"""
MIL-NCE baseline (论文里的 MIL-NCE / Multi-Instance InfoNCE)

- 训练：每篇文章对应一个 image-bag（固定 5 张图）
        对 batch 内所有 (text_j, bag_k) 计算 logits：
            logits[j,k] = logsumexp_i( logit_scale * <t_j, img_{k,i}> )
        然后做 CLIP 式双向对比学习（text->bag + bag->text）

- 测试：按论文评测写法，用 mean(sim) 做 bag 分数：
        score(text, bag) = mean_i <t, img_i>
      计算 R@1/5/10、Median Rank

依赖：
  pip install ftfy regex tqdm
  pip install git+https://github.com/openai/CLIP.git
"""

import math
import random
from typing import List, Tuple, Dict, Any
from utils import set_seed

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import clip
from dataset import CaptioningDataset, captioning_collate_fn


# ===================== 你只需要改这些参数 =====================
# 数据
TRAIN_JSON_PATH = "data/captioning_dataset_5imgs_train_60.json"
TEST_JSON_PATH  = "data/captioning_dataset_5imgs_test_40.json"
IMAGE_ROOT = "data/resized"

# CLIP
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = True

# 论文里会扩到 256（需要改 positional embedding + attn_mask + 每个block的mask）
# 不想折腾就用 77（更稳）。想贴论文就用 256。
TEXT_CONTEXT_LENGTH = 256  # 77 或 256
TEXT_TRUNCATE = True

# 训练
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 16
PIN_MEMORY = True

EPOCHS = 15
LR = 2e-5
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 20000  # 数据小会自动缩短
GRAD_CLIP_NORM = 1.0
PRINT_EVERY = 20


CLIP_MODEL_NAME = "ViT-L/14@336px"
SAVE_NAME = CLIP_MODEL_NAME.replace("/","").replace("-","")
SAVE_PATH = f"./ckpt/mil_nce_clip_{SAVE_NAME}.pt"
# 测试评测分块
CHUNK_SIZE = 512
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
    - 每个 transformer block 的 attn_mask（否则仍是 77x77 会报错）
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
    """
    贴论文：只 finetune projection layers（+ logit_scale）
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
    关键：openai/CLIP 在 CUDA 上很多参数是 fp16，GradScaler 会报
      "Attempting to unscale FP16 gradients."
    所以把 requires_grad=True 的参数转成 fp32。
    """
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()


def build_optimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)


@torch.no_grad()
def compute_test_embeddings_mean_pool(
    dataloader: DataLoader,
    model,
    device: torch.device,
    use_fp16: bool,
    require_num_images: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    测试：按论文评测写法，score 用 mean(sim)。
    这里预先算：
      text_feats: [N,D] (L2 norm)
      bag_feats : [N,D] = mean over 5 images of (L2-normed image feats) (no renorm)
    """
    model.eval()
    all_text, all_bag = [], []

    amp_enabled = (device.type == "cuda" and use_fp16)
    autocast = lambda: torch.amp.autocast("cuda", enabled=amp_enabled) if device.type == "cuda" \
                       else torch.amp.autocast("cpu", enabled=False)

    for batch in tqdm(dataloader, desc="Test Embedding"):
        articles: List[str] = batch["articles"]
        imgs_list: List[torch.Tensor] = batch["images"]  # list of [5,3,H,W]

        for t in imgs_list:
            if t.ndim != 4 or t.shape[0] != require_num_images:
                raise ValueError(f"Expect {require_num_images} imgs per sample, got {tuple(t.shape)}")

        tokens = clip.tokenize(
            articles,
            context_length=model.context_length,
            truncate=TEXT_TRUNCATE
        ).to(device)

        imgs = torch.stack(imgs_list, dim=0)  # [B,5,3,H,W]
        bsz, k, c, h, w = imgs.shape
        imgs = imgs.view(bsz * k, c, h, w).to(device)

        with autocast():
            txt = model.encode_text(tokens)      # [B,D]
            img = model.encode_image(imgs)       # [B*5,D]

        txt = txt.float()
        img = img.float()
        txt = txt / txt.norm(dim=-1, keepdim=True)
        img = img / img.norm(dim=-1, keepdim=True)

        img = img.view(bsz, k, -1)              # [B,5,D]
        bag = img.mean(dim=1)                   # [B,D] (no renorm)

        all_text.append(txt.cpu())
        all_bag.append(bag.cpu())

    return torch.cat(all_text, dim=0), torch.cat(all_bag, dim=0)


@torch.no_grad()
def retrieval_metrics(
    text_feats: torch.Tensor,
    bag_feats: torch.Tensor,
    device: torch.device,
    chunk_size: int = 512,
) -> Dict[str, float]:
    """
    GT：第 i 个 text 对应第 i 个 bag（test_loader 必须 shuffle=False）
    """
    N, _ = text_feats.shape
    text_feats = text_feats.to(device)
    bag_feats = bag_feats.to(device)
    bag_T = bag_feats.t()

    ranks = torch.empty(N, dtype=torch.long, device="cpu")

    for start in tqdm(range(0, N, chunk_size), desc="Test Scoring"):
        end = min(start + chunk_size, N)
        scores = text_feats[start:end] @ bag_T  # [C,N]

        gt_idx = torch.arange(start, end, device=device)
        row_idx = torch.arange(0, end - start, device=device)
        gt_scores = scores[row_idx, gt_idx]

        r = (scores > gt_scores.unsqueeze(1)).sum(dim=1) + 1
        ranks[start:end] = r.cpu()

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

    # 1) Load CLIP + preprocess
    model, preprocess = clip.load(CLIP_MODEL_NAME, device=device, jit=False)

    # 2) 可选：扩 context length
    if TEXT_CONTEXT_LENGTH != model.context_length:
        extend_clip_context_length(model, TEXT_CONTEXT_LENGTH)

    # 3) 只训练 projection layers（贴论文）
    freeze_to_projection_only(model)
    cast_trainable_params_to_fp32(model)  # ✅ 避免 GradScaler unscale FP16 grads 报错
    optimizer = build_optimizer(model)

    # 4) Datasets / Loaders
    train_ds = CaptioningDataset(
        json_path=TRAIN_JSON_PATH,
        image_root=IMAGE_ROOT,
        transform=preprocess,
        use_headline=False,
    )
    test_ds = CaptioningDataset(
        json_path=TEST_JSON_PATH,
        image_root=IMAGE_ROOT,
        transform=preprocess,
        use_headline=False,
    )

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
    warmup_steps = min(WARMUP_STEPS, max(0, total_steps // 10))  # 数据小自动缩短
    print(f"Train size={len(train_ds)}, Test size={len(test_ds)}")
    print(f"Total steps={total_steps}, Warmup steps={warmup_steps}")
    print(f"CLIP ctx={model.context_length}")

    amp_enabled = (device.type == "cuda" and USE_FP16)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None
    autocast = lambda: torch.amp.autocast("cuda", enabled=amp_enabled) if device.type == "cuda" \
                    else torch.amp.autocast("cpu", enabled=False)

    global_step = 0
    model.train()

    for epoch in range(1, EPOCHS + 1):
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            articles: List[str] = batch["articles"]
            imgs_list: List[torch.Tensor] = batch["images"]  # list of [5,3,H,W]

            for t in imgs_list:
                if t.ndim != 4 or t.shape[0] != REQUIRE_NUM_IMAGES:
                    raise ValueError(f"Expect {REQUIRE_NUM_IMAGES} imgs per sample, got {tuple(t.shape)}")

            # tokens
            tokens = clip.tokenize(
                articles,
                context_length=model.context_length,
                truncate=TEXT_TRUNCATE
            ).to(device)

            # images: [B,5,3,H,W] -> [B*5,3,H,W]
            imgs = torch.stack(imgs_list, dim=0)
            bsz, k, c, h, w = imgs.shape
            imgs = imgs.view(bsz * k, c, h, w).to(device)

            # lr schedule
            lr_scale = cosine_with_warmup_lr(global_step, total_steps, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = LR * lr_scale

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                # encode
                txt = model.encode_text(tokens)       # [B,D]
                img = model.encode_image(imgs)        # [B*K,D]

                # normalize (float)
                txt = txt.float()
                img = img.float()
                txt = txt / txt.norm(dim=-1, keepdim=True)             # [B,D]
                img = img / img.norm(dim=-1, keepdim=True)             # [B*K,D]
                img = img.view(bsz, k, -1)                             # [B,5,D]

                # logits per-image: [text=B, bag=B, k]
                # sim = dot(txt_j, img_{bag,i})
                per_img_logits = torch.einsum("td,bkd->tbk", txt, img)  # [B,B,5]
                logit_scale = model.logit_scale.exp().float()
                per_img_logits = per_img_logits * logit_scale          # [B,B,5]

                # MIL-NCE bag logits: log sum exp over instances
                bag_logits = torch.logsumexp(per_img_logits, dim=-1)     # [B,B]

                labels = torch.arange(bsz, device=device)
                loss_t2b = F.cross_entropy(bag_logits, labels)
                loss_b2t = F.cross_entropy(bag_logits.t(), labels)
                loss = (loss_t2b + loss_b2t) / 2.0

            if device.type == "cuda" and scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        GRAD_CLIP_NORM
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        GRAD_CLIP_NORM
                    )
                optimizer.step()

            global_step += 1
            if global_step % PRINT_EVERY == 0:
                pbar.set_postfix(loss=float(loss.item()), lr=optimizer.param_groups[0]["lr"])

    # 5) Save

    torch.save(
        {
            "model": model.state_dict(),
            "clip_model_name": CLIP_MODEL_NAME,
            "text_context_length": model.context_length,
        },
        SAVE_PATH
    )
    print(f"[Saved] {SAVE_PATH}")

    # 6) Test (mean(sim) scoring)
    model.eval()
    text_feats, bag_feats = compute_test_embeddings_mean_pool(
        test_loader, model, device,
        use_fp16=USE_FP16,
        require_num_images=REQUIRE_NUM_IMAGES,
    )
    metrics = retrieval_metrics(text_feats, bag_feats, device=device, chunk_size=CHUNK_SIZE)

    print(f"\n==== Test Results (MIL-NCE trained, mean(sim) eval) using CLIP {CLIP_MODEL_NAME} ====")
    print(f"N={int(metrics['N'])}")
    print(f"R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}")
    print(f"MedianRank={int(metrics['MedianRank'])}")


if __name__ == "__main__":
    main()
