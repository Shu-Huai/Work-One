import os
import json
from typing import List, Dict, Any

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CaptioningDataset(Dataset):
    """
    每个样本：
        {
            "article_id": str,
            "article": str,
            "images": List[Tensor],      # 每张图片一个 tensor，形状 [3, H, W]
            "captions": List[str],       # 每张图片对应的 caption
        }
    """
    def __init__(
        self,
        json_path: str,
        image_root: str,
        transform=None,
        use_headline: bool = False,
    ):
        """
        :param json_path: captioning_dataset.json 路径
        :param image_root: 已经 resize 好的图像根目录
        :param transform: 对 PIL Image 的变换（如 ToTensor / Normalize）
        :param use_headline: 如果为 True，则将 headline.main 拼到 article 前面
        """
        self.json_path = json_path
        self.image_root = image_root
        self.transform = transform

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)  # 顶层是 dict: {article_id: value_dict}

        self.samples: List[Dict[str, Any]] = []

        # 为了顺序稳定，按 key 排序（可选）
        for article_id, value in sorted(data.items(), key=lambda x: x[0]):
            # 文章正文
            article_text = value.get("article", "")

            if use_headline:
                headline = value.get("headline", {})
                main_headline = headline.get("main", "")
                if main_headline:
                    # 也可以根据需要加分隔符 \n\n 等
                    article_text = main_headline + "\n\n" + article_text

            images_dict = value.get("images", {})
            img_entries = []

            # images_dict: {"0": "caption0", "1": "caption1", ...}
            for idx_str, caption in images_dict.items():
                # 构造图像文件名：key_编号.jpg
                filename = f"{article_id}_{idx_str}.jpg"
                img_path = os.path.join(image_root, filename)
                img_entries.append(
                    {
                        "path": img_path,
                        "caption": caption,
                        "idx": int(idx_str),
                    }
                )

            # 按编号排序，保证顺序是 0,1,2,...
            img_entries.sort(key=lambda x: x["idx"])

            # 也可以在这里过滤掉文件不存在的情况
            # img_entries = [e for e in img_entries if os.path.exists(e["path"])]

            self.samples.append(
                {
                    "article_id": article_id,
                    "article": article_text,
                    "images_meta": img_entries,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        article_id = sample["article_id"]
        article_text = sample["article"]
        images_meta = sample["images_meta"]

        images: List[torch.Tensor] = []
        captions: List[str] = []

        for meta in images_meta:
            img_path = meta["path"]
            caption = meta["caption"]

            # 读取并转成 RGB
            img = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)

            images.append(img)
            captions.append(caption)

        return {
            "article_id": article_id,
            "article": article_text,
            "images": images,      # List[Tensor]
            "captions": captions,  # List[str]，顺序和 images 对应
        }



def captioning_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    batch 是若干个 __getitem__ 的返回结果组成的列表。
    返回：
        {
            "article_ids": List[str],
            "articles": List[str],
            "images": List[Tensor],       # len = batch_size，每个元素 shape [num_imgs_i, 3, H, W]
            "captions": List[List[str]],  # len = batch_size，每个元素是该文章所有 caption 的列表
        }
    """
    article_ids = [item["article_id"] for item in batch]
    articles = [item["article"] for item in batch]

    # 每篇文章内部可以 stack 成 [num_imgs, 3, H, W]
    imgs_per_article = [
        torch.stack(item["images"], dim=0)  # 形状 [num_imgs_i, 3, H, W]
        for item in batch
    ]
    captions_per_article = [item["captions"] for item in batch]

    return {
        "article_ids": article_ids,
        "articles": articles,
        "images": imgs_per_article,
        "captions": captions_per_article,
    }



