"""
Pretrained CLIP baseline (GoodNews):
- Query: article text
- Candidate: each article's image-set (fixed 5 images)
- Score(article, image_set) = mean_i cosine_sim(text, image_i)
- Metrics: R@1/5/10, Median Rank
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import clip  # pip install git+https://github.com/openai/CLIP.git

from dataset import CaptioningDataset, captioning_collate_fn


# ===================== 你只需要改这些参数 =====================
REQUIRE_NUM_IMAGES = 4   # 你的数据已保证每条 5 张图
JSON_PATH = f"data/captioning_dataset_{REQUIRE_NUM_IMAGES}imgs_test_40.json"
IMAGE_ROOT = "data/resized"

CLIP_MODEL_NAME = "ViT-L/14@336px"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
NUM_WORKERS = 8
PIN_MEMORY = True

TEXT_TRUNCATE = True     # CLIP 默认 max_len=77，truncate=True 会截断超长 article
CHUNK_SIZE = 512         # 分块算相似度，显存/内存不够就调小
USE_FP16 = True          # GPU上建议开；CPU会自动无效
# ============================================================


@torch.no_grad()
def compute_pretrained_clip_embeddings(
    dataloader: DataLoader,
    model,
    device: torch.device,
    require_num_images: int = 5,
    use_fp16: bool = True,
    text_truncate: bool = True,
):
    """
    输出：
      text_feats:   [N, D]  (L2 norm)
      imgset_feats: [N, D]  = mean over K images of (L2-normed image feats), 不再二次归一化
    这样 score = text @ imgset_feats^T 等价于 mean_i dot(text, img_i)
    """
    model.eval()

    all_text = []
    all_imgset = []

    use_cuda_amp = (device.type == "cuda" and use_fp16)
    autocast_device = "cuda" if use_cuda_amp else "cpu"

    for batch in tqdm(dataloader, desc="Embedding"):
        articles = batch["articles"]              # List[str]
        imgs_list = batch["images"]              # List[Tensor], each: [K,3,H,W]

        # 运行时检查：每条必须是 K=5
        for t in imgs_list:
            if t.ndim != 4 or t.shape[0] != require_num_images:
                raise ValueError(f"Expect each sample has {require_num_images} images, but got shape={tuple(t.shape)}")

        # text
        tokens = clip.tokenize(articles, truncate=text_truncate).to(device)

        # images: flatten BxK -> (B*K)
        imgs = torch.stack(imgs_list, dim=0)     # [B,K,3,H,W]
        bsz, k, c, h, w = imgs.shape
        imgs = imgs.view(bsz * k, c, h, w).to(device)

        with torch.amp.autocast(autocast_device):
            text_feats = model.encode_text(tokens)   # [B, D]
            img_feats = model.encode_image(imgs)     # [B*K, D]

        # normalize
        text_feats = text_feats.float()
        img_feats = img_feats.float()
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        # reshape back and mean-pool (NO renorm)
        img_feats = img_feats.view(bsz, k, -1)       # [B,K,D]
        imgset_feats = img_feats.mean(dim=1)         # [B,D]

        all_text.append(text_feats.cpu())
        all_imgset.append(imgset_feats.cpu())

    return torch.cat(all_text, dim=0), torch.cat(all_imgset, dim=0)


@torch.no_grad()
def retrieval_metrics(
    text_feats: torch.Tensor,      # [N,D] normalized
    imgset_feats: torch.Tensor,    # [N,D] mean of normalized image feats (not renorm)
    device: torch.device,
    chunk_size: int = 512,
):
    """
    rank（1=最好），并给出 R@1/5/10 和 median rank
    默认“第 i 个文本”的 GT 就是“第 i 个 image-set”（与你的数据组织一致）
    """
    N, _ = text_feats.shape
    text_feats = text_feats.to(device)
    imgset_feats = imgset_feats.to(device)
    imgset_T = imgset_feats.t()  # [D,N]

    ranks = torch.empty(N, dtype=torch.long, device="cpu")

    for start in tqdm(range(0, N, chunk_size), desc="Scoring"):
        end = min(start + chunk_size, N)
        txt = text_feats[start:end]         # [C,D]
        scores = txt @ imgset_T             # [C,N]

        gt_idx = torch.arange(start, end, device=device)
        row_idx = torch.arange(0, end - start, device=device)
        gt_scores = scores[row_idx, gt_idx]

        r = (scores > gt_scores.unsqueeze(1)).sum(dim=1) + 1
        ranks[start:end] = r.cpu()

    r1 = (ranks <= 1).float().mean().item()
    r5 = (ranks <= 5).float().mean().item()
    r10 = (ranks <= 10).float().mean().item()
    median_rank = ranks.median().item()

    return {"N": N, "R@1": r1, "R@5": r5, "R@10": r10, "MedianRank": median_rank}


def main():
    device = torch.device(DEVICE)

    # Load pretrained CLIP + its preprocess
    model, preprocess = clip.load(CLIP_MODEL_NAME, device=device, jit=False)

    # Use your dataset
    ds = CaptioningDataset(
        json_path=JSON_PATH,
        image_root=IMAGE_ROOT,
        transform=preprocess,   # 直接用 CLIP 自带预处理
        use_headline=False,     # 文本用 article
    )
    print(f"Loaded dataset size: {len(ds)}")

    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,  # 必须 false，保证 i 对齐 i（GT=同 index）
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY if device.type == "cuda" else False,
        collate_fn=captioning_collate_fn,
        drop_last=False,
    )

    text_feats, imgset_feats = compute_pretrained_clip_embeddings(
        dl, model, device,
        require_num_images=REQUIRE_NUM_IMAGES,
        use_fp16=USE_FP16,
        text_truncate=TEXT_TRUNCATE,
    )

    metrics = retrieval_metrics(
        text_feats=text_feats,
        imgset_feats=imgset_feats,
        device=device,
        chunk_size=CHUNK_SIZE,
    )

    print(f"\n==== Pretrained CLIP {CLIP_MODEL_NAME} Results ====")
    print(f"N = {metrics['N']}")
    print(f"R@1  = {metrics['R@1']:.4f}")
    print(f"R@5  = {metrics['R@5']:.4f}")
    print(f"R@10 = {metrics['R@10']:.4f}")
    print(f"Median Rank = {metrics['MedianRank']}")


if __name__ == "__main__":
    main()
