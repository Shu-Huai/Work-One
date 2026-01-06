
import os
import json
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import clip  # pip install ftfy regex tqdm; pip install git+https://github.com/openai/CLIP.git
from dataset import CaptioningDataset, captioning_collate_fn

def compute_clip_embeddings(
    dataloader: DataLoader,
    model,
    device: torch.device,
) -> (torch.Tensor, torch.Tensor):
    """
    返回：
        text_embs:  [N, D]  每篇 article 一个向量
        imgset_embs: [N, D] 每个 image-set (该 article 对应的所有图的平均) 一个向量
    """
    model.eval()

    all_text_embs = []
    all_imgset_embs = []

    with torch.no_grad():
        # 加 tqdm
        for batch in tqdm(dataloader, desc="Computing CLIP embeddings"):
            articles: List[str] = batch["articles"]
            images_per_article: List[torch.Tensor] = batch["images"]

            # ----- 文本编码 -----
            text_tokens = clip.tokenize(articles, truncate=True).to(device)  # [B, 77]
            text_features = model.encode_text(text_tokens)  # [B, D]

            # ----- 图像集合编码 -----
            imgset_features = []
            for imgs in images_per_article:
                # imgs: [num_imgs_i, 3, H, W]
                imgs = imgs.to(device)
                img_feats = model.encode_image(imgs)  # [num_imgs_i, D]
                img_feats = img_feats.mean(dim=0, keepdim=True)  # [1, D]，多图取平均
                imgset_features.append(img_feats)

            imgset_features = torch.cat(imgset_features, dim=0)  # [B, D]

            all_text_embs.append(text_features.cpu())
            all_imgset_embs.append(imgset_features.cpu())

    text_embs = torch.cat(all_text_embs, dim=0)      # [N, D]
    imgset_embs = torch.cat(all_imgset_embs, dim=0)  # [N, D]

    # L2 normalize
    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
    imgset_embs = imgset_embs / imgset_embs.norm(dim=-1, keepdim=True)

    return text_embs, imgset_embs


def compute_recall_at_k(sim_matrix: torch.Tensor, ks=(1, 5, 10)):
    """
    sim_matrix: [N, N]，第 i 行是 query 第 i 篇 article
    假设第 i 篇 article 的正例 image-set 就是第 i 个（同 index）
    """
    N = sim_matrix.size(0)
    ranks = sim_matrix.argsort(dim=1, descending=True)  # [N, N]

    device = sim_matrix.device
    target = torch.arange(N, device=device)  # [N]

    recalls = {}
    for k in ks:
        # 看每一行前 k 个里有没有自己的 index
        correct = (ranks[:, :k] == target.unsqueeze(1)).any(dim=1)  # [N]
        recalls[f"R@{k}"] = correct.float().mean().item()
    return recalls


def main():
    json_path = "data/captioning_dataset_5imgs.json"
    image_root = "data/resized"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) 加载预训练 CLIP 模型和它自带的图像预处理
    model, preprocess = clip.load("ViT-L/14", device=device)  # 或 RN50 等其它 backbone

    # 2) 构造 Dataset / DataLoader
    dataset = CaptioningDataset(
        json_path=json_path,
        image_root=image_root,
        transform=preprocess,   # 直接用 CLIP 的预处理
        use_headline=False,     # 你要用 headline 的话改成 True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,           # 看你显存调整
        shuffle=False,
        num_workers=16,
        collate_fn=captioning_collate_fn,
    )

    # 3) 前向推理得到 embeddings（不训练）
    text_embs, imgset_embs = compute_clip_embeddings(dataloader, model, device)

    # 4) 计算 article -> image-set 的相似度矩阵
    #    sim[i, j] = 文本 i 与 image-set j 的余弦相似度
    sim_matrix = text_embs @ imgset_embs.t()  # [N, N]

    # 5) 计算 Recall@K
    recalls = compute_recall_at_k(sim_matrix, ks=(1, 5, 10))
    for k, v in recalls.items():
        print(f"{k}: {v * 100:.2f}%")

    print("Done.")


if __name__ == "__main__":
    main()