"""
Single Image baseline (论文里的 Single Image-Text Contrastive / InfoNCE)

- 训练集：captioning_dataset_5imgs_train.json
- 测试集：captioning_dataset_5imgs_test.json
- 训练：每篇文章随机取 1 张图，做 CLIP 对比学习（batch 内互为负样本）
- 测试：对 5 张图分别算 sim(text, img)，再取平均作为 image-set 分数
- 指标：R@1/5/10、Median Rank

依赖：
  pip install ftfy regex tqdm
  pip install git+https://github.com/openai/CLIP.git
"""

import math
import random
from typing import List, Dict, Tuple
import torch.amp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import set_all_seeds, cosine_with_warmup_lr, extend_clip_context_length, freeze_to_projection_only
from metrics.retrieval_metrics import retrieval_metrics_multi
import clip

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

# 论文里会扩到 256（需要改 positional embedding）
# 不想动结构就保持 77（更稳）。你要贴论文训练设置就用 256。
TEXT_CONTEXT_LENGTH = 256  # 77 或 256
TEXT_TRUNCATE = True      # 超长 article 截断

# 训练
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 8
PIN_MEMORY = True

EPOCHS = 10
LR = 2e-5
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 20000      # 数据小会自动缩短
GRAD_CLIP_NORM = 1.0

PRINT_EVERY = 50          # 每多少 step 打印一次 loss/lr
SAVE_NAME = CLIP_MODEL_NAME.replace("/","").replace("-","")
SAVE_PATH = f"./ckpt/single_image_clip_{SAVE_NAME}.pt"

# 测试评测分块
TEXT_CHUNK = 32
CHUNK_SIZE = 512
# ============================================================

def build_optimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)


@torch.no_grad()
def compute_embeddings_for_eval(
    dataloader: DataLoader,
    model,
    device: torch.device,
    text_context_length: int,
    text_truncate: bool,
    use_fp16: bool,
    require_num_images: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    输出：
      text_feats:   [N, D]  (L2 norm)
      imgset_feats: [N, D]  = mean over 5 images of (L2-normed image feats), 不再二次归一化
    """
    model.eval()
    all_text, all_imgset = [], []

    use_cuda_amp = (device.type == "cuda" and use_fp16)

    for batch in tqdm(dataloader, desc="Test Embedding"):
        articles: List[str] = batch["articles"]
        imgs_list: List[torch.Tensor] = batch["images"]  # list of [5,3,H,W]

        for t in imgs_list:
            if t.ndim != 4 or t.shape[0] != require_num_images:
                raise ValueError(f"Expect each sample has {require_num_images} images, but got {tuple(t.shape)}")

        tokens = clip.tokenize(
            articles,
            context_length=text_context_length,
            truncate=text_truncate
        ).to(device)

        imgs = torch.stack(imgs_list, dim=0)  # [B,5,3,H,W]
        bsz, k, c, h, w = imgs.shape
        imgs = imgs.view(bsz * k, c, h, w).to(device)

        with torch.amp.autocast("cuda" if use_cuda_amp else "cpu"):
            text_feats = model.encode_text(tokens)  # [B,D]
            img_feats = model.encode_image(imgs)    # [B*5,D]

        text_feats = text_feats.float()
        img_feats = img_feats.float()
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        img_feats = img_feats.view(bsz, k, -1)     # [B,5,D]
        imgset_feats = img_feats.mean(dim=1)       # [B,D]

        all_text.append(text_feats.cpu())
        all_imgset.append(imgset_feats.cpu())

    return torch.cat(all_text, dim=0), torch.cat(all_imgset, dim=0)


def main():
    set_all_seeds(SEED)
    device = torch.device(DEVICE)

    # 1) Load CLIP + preprocess
    model, preprocess = clip.load(CLIP_MODEL_NAME, device=device, jit=False)

    # 2) 可选：扩 context length
    if TEXT_CONTEXT_LENGTH != model.context_length:
        extend_clip_context_length(model, TEXT_CONTEXT_LENGTH)

    # 3) 只训练 projection layers（贴论文）
    freeze_to_projection_only(model)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
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

    # 5) Scheduler（cosine + warmup）
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = min(WARMUP_STEPS, max(0, total_steps // 10))  # 数据小自动缩短
    print(f"Train size={len(train_ds)}, Test size={len(test_ds)}")
    print(f"Total steps={total_steps}, Warmup steps={warmup_steps}")
    scaler = torch.amp.GradScaler(DEVICE)

    global_step = 0
    model.train()

    for epoch in range(1, EPOCHS + 1):
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            articles: List[str] = batch["articles"]
            imgs_list: List[torch.Tensor] = batch["images"]  # list of [5,3,H,W]

            # Single Image：每篇文章随机取 1 张图
            picked = []
            for imgs in imgs_list:
                if imgs.ndim != 4 or imgs.shape[0] != REQUIRE_NUM_IMAGES:
                    raise ValueError(f"Expect {REQUIRE_NUM_IMAGES} imgs, got {tuple(imgs.shape)}")
                j = random.randint(0, REQUIRE_NUM_IMAGES - 1)
                picked.append(imgs[j])
            images = torch.stack(picked, dim=0).to(device)  # [B,3,H,W]

            tokens = clip.tokenize(
                articles,
                context_length=TEXT_CONTEXT_LENGTH,
                truncate=TEXT_TRUNCATE
            ).to(device)

            lr_scale = cosine_with_warmup_lr(global_step, total_steps, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = LR * lr_scale

            optimizer.zero_grad(set_to_none=True)

            use_cuda_amp = (device.type == "cuda" and USE_FP16)

            with torch.amp.autocast("cuda" if use_cuda_amp else "cpu"):
                img_feats = model.encode_image(images)
                txt_feats = model.encode_text(tokens)

                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

                logit_scale = model.logit_scale.exp()
                logits_i2t = logit_scale * (img_feats @ txt_feats.t())
                logits_t2i = logits_i2t.t()

                labels = torch.arange(images.size(0), device=device)
                loss = (F.cross_entropy(logits_i2t, labels) + F.cross_entropy(logits_t2i, labels)) / 2.0

            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    GRAD_CLIP_NORM
                )
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            if global_step % PRINT_EVERY == 0:
                pbar.set_postfix(loss=float(loss.item()), lr=optimizer.param_groups[0]["lr"])

    # 6) Save
    torch.save(
        {
            "model": model.state_dict(),
            "clip_model_name": CLIP_MODEL_NAME,
            "text_context_length": TEXT_CONTEXT_LENGTH,
        },
        SAVE_PATH
    )
    print(f"[Saved] {SAVE_PATH}")

    # 7) Test
    model.eval()
    text_feats, imgset_feats = compute_embeddings_for_eval(
        test_loader, model, device,
        text_context_length=TEXT_CONTEXT_LENGTH,
        text_truncate=TEXT_TRUNCATE,
        use_fp16=USE_FP16,
        require_num_images=REQUIRE_NUM_IMAGES,
    )
    metrics = retrieval_metrics_multi(
        text_embs_cpu=text_feats.detach().cpu(),
        imgset_embs_cpu=imgset_feats.detach().cpu(),
        logit_scale=model.logit_scale,      # 或 clip_model.logit_scale
        device=device,
        text_chunk=TEXT_CHUNK,              # 原来的 chunk_size 用这个
        cand_chunk=CHUNK_SIZE,         # 你需要新增一个，常用 1024/2048/4096
    )
    print(f"\n==== Test Results (Single Image) on {CLIP_MODEL_NAME} ====")
    print(f"N={int(metrics['N'])}")
    print(f"R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}")
    print(f"MedianRank={int(metrics['MedianRank'])}")


if __name__ == "__main__":
    main()
