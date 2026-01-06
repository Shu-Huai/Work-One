import os
import json
import math
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

import clip  # pip install ftfy regex tqdm; pip install git+https://github.com/openai/CLIP.git


# ===================== Dataset：单图 + 文本 =====================

class SingleImageCaptionDataset(Dataset):
    """
    每个样本是一张图 + 一个文本：
        {
            "image": Tensor[3,H,W],
            "text": str,
            "article_id": str,
            "image_idx": int,
        }

    text_source:
        - "article": 用整篇 article 文本
        - "caption": 用该图自己的 caption
        - "article+caption": caption + "\n\n" + article
    """
    def __init__(
        self,
        json_path: str,
        image_root: str,
        transform=None,
        text_source: str = "article",
        use_headline: bool = False,
    ):
        super().__init__()
        self.json_path = json_path
        self.image_root = image_root
        self.transform = transform
        self.text_source = text_source
        self.use_headline = use_headline

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)  # 顶层: {article_id: value_dict}

        self.samples: List[Dict[str, Any]] = []

        # 为了顺序稳定，按 article_id 排序
        for article_id, value in sorted(data.items(), key=lambda x: x[0]):
            # 构造 article 文本
            article_text = value.get("article", "")
            if use_headline:
                headline = value.get("headline", {})
                main_headline = headline.get("main", "")
                if main_headline:
                    article_text = main_headline + "\n\n" + article_text

            images_dict = value.get("images", {})  # {"0": caption0, "1": caption1, ...}

            # 遍历每一张图，展开成单独样本
            for idx_str, caption in images_dict.items():
                idx_int = int(idx_str)
                filename = f"{article_id}_{idx_str}.jpg"
                img_path = os.path.join(image_root, filename)

                # 决定文本内容
                if self.text_source == "article":
                    text = article_text
                elif self.text_source == "caption":
                    text = caption
                elif self.text_source == "article+caption":
                    text = caption + "\n\n" + article_text
                else:
                    raise ValueError(f"Unknown text_source: {self.text_source}")

                self.samples.append(
                    {
                        "image_path": img_path,
                        "text": text,
                        "article_id": article_id,
                        "image_idx": idx_int,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        img_path = sample["image_path"]
        text = sample["text"]

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return {
            "image": img,
            "text": text,
            "article_id": sample["article_id"],
            "image_idx": sample["image_idx"],
        }


def single_image_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # images: [B, 3, H, W]
    images = torch.stack([item["image"] for item in batch], dim=0)
    texts = [item["text"] for item in batch]
    article_ids = [item["article_id"] for item in batch]
    image_indices = [item["image_idx"] for item in batch]

    return {
        "images": images,
        "texts": texts,
        "article_ids": article_ids,
        "image_indices": image_indices,
    }


# ===================== CLIP 文本长度扩展到 256 =====================

def extend_clip_text_context_length(model: "clip.model.CLIP", new_context_length: int = 256):
    """
    将 CLIP 的 context_length 从 77 扩展到 256：
        - 扩大 positional_embedding 到 [new_ctx, dim]
        - 更新 model.context_length
        - 为所有 Transformer block 重建 attn_mask (new_ctx x new_ctx)
    """
    old_ctx = model.context_length
    if new_context_length <= old_ctx:
        # 已经够长，就不动
        return model

    with torch.no_grad():
        # ---- 1) 扩展位置编码 ----
        old_pos_embed = model.positional_embedding  # [old_ctx, dim]
        dim = old_pos_embed.size(1)
        device = old_pos_embed.device
        dtype = old_pos_embed.dtype

        new_pos_embed = torch.empty(new_context_length, dim, device=device, dtype=dtype)
        # 复制旧的
        new_pos_embed[:old_ctx] = old_pos_embed
        # 新增部分随机初始化
        nn.init.normal_(new_pos_embed[old_ctx:], std=0.01)

        model.positional_embedding = nn.Parameter(new_pos_embed)
        model.context_length = new_context_length

        # ---- 2) 重建 attention mask，并写回每一层 ResidualAttentionBlock ----
        # 等价于 CLIP 的 build_attention_mask()
        attn_mask = torch.empty(new_context_length, new_context_length, device=device)
        attn_mask.fill_(float("-inf"))
        attn_mask.triu_(1)  # 上三角（不含对角）为 -inf，下三角 & 对角为 0

        # 把这个 mask 塞进所有 resblocks
        if hasattr(model, "transformer") and hasattr(model.transformer, "resblocks"):
            for block in model.transformer.resblocks:
                # 官方实现里 block 是 ResidualAttentionBlock，有 .attn_mask 属性
                if hasattr(block, "attn_mask"):
                    block.attn_mask = attn_mask

    return model



# ===================== 只微调 projection + logit_scale + pos_embed =====================

def set_trainable_params_for_single_image(model: "clip.model.CLIP"):
    """
    冻结大部分参数，只保留：
        - visual.proj (图像 projection)
        - text_projection (文本 projection)
        - logit_scale
        - positional_embedding（尤其是新增的部分，需要学习）
    """
    # 先全部冻结
    for p in model.parameters():
        p.requires_grad = False

    # 1) 文本侧 projection: model.text_projection (nn.Parameter [D,D] or [D,dim_text])
    if hasattr(model, "text_projection") and model.text_projection is not None:
        model.text_projection.requires_grad = True

    # 2) 图像侧 projection: model.visual.proj
    if hasattr(model, "visual") and hasattr(model.visual, "proj"):
        proj = model.visual.proj
        if proj is not None:
            if isinstance(proj, nn.Parameter):
                proj.requires_grad = True
            else:
                for p in proj.parameters():
                    p.requires_grad = True

    # 3) logit_scale
    if hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad = True

    # 4) 文本位置编码（77->256 的新增部分也要学习）
    if hasattr(model, "positional_embedding"):
        model.positional_embedding.requires_grad = True

    # 打印一下可训练参数数量，方便你确认
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total/1e6:.2f}M, Trainable: {trainable/1e6:.2f}M")

    return model


# ===================== 学习率调度：20k warmup + cosine =====================

def get_cosine_schedule_with_warmup_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int = 20000,
) -> float:
    """
    step: 当前步数（从 1 开始）
    total_steps: 训练总步数（用于 cosine）
    base_lr: 最大学习率
    warmup_steps: 线性 warmup 步数
    """
    if step < warmup_steps:
        # 线性从 0 -> base_lr
        return base_lr * float(step) / float(max(1, warmup_steps))

    # 余弦退火，step >= warmup_steps
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr


# ===================== 训练循环：Single Image InfoNCE =====================

def train_single_image(
    json_path: str = "data/captioning_dataset.json",
    image_root: str = "data/resized",
    batch_size: int = 32,
    num_epochs: int = 1,
    base_lr: float = 1e-5,
    warmup_steps: int = 20000,
    max_text_len: int = 256,
    text_source: str = "article",  # "article" / "caption" / "article+caption"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) 加载预训练 CLIP ViT-L/14
    model, preprocess = clip.load("ViT-L/14", device=device)
    print("Loaded CLIP ViT-L/14.")

    # 2) 扩展文本长度到 256
    model = extend_clip_text_context_length(model, new_context_length=max_text_len)
    print(f"Extended CLIP context_length to {model.context_length}.")

    # 3) 设置哪些参数需要训练
    model = set_trainable_params_for_single_image(model)

    # 4) 构造 Dataset / DataLoader
    dataset = SingleImageCaptionDataset(
        json_path=json_path,
        image_root=image_root,
        transform=preprocess,
        text_source=text_source,
        use_headline=False,  # 你要用 headline 再改 True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 先用 0，确认没问题再开多进程
        collate_fn=single_image_collate_fn,
    )

    steps_per_epoch = math.ceil(len(dataloader))
    total_steps = steps_per_epoch * num_epochs
    print(f"Dataset size: {len(dataset)}, steps/epoch: {steps_per_epoch}, total_steps: {total_steps}")

    # 5) 优化器（只传入 requires_grad=True 的参数）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=base_lr,
        betas=(0.9, 0.98),  # 近似 CLIP 训练配置
        eps=1e-6,
        weight_decay=0.0,   # 论文没明确写，可按需调整
    )

    global_step = 0
    model.train()
    losses = []
    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")
        pbar = tqdm(dataloader, desc=f"Training epoch {epoch + 1}")
        batch_losses = []
        for batch in pbar:
            global_step += 1

            images: torch.Tensor = batch["images"].to(device)  # [B, 3, H, W]
            texts: List[str] = batch["texts"]

            # 更新学习率（手动 scheduler）
            lr = get_cosine_schedule_with_warmup_lr(
                step=global_step,
                total_steps=total_steps,
                base_lr=base_lr,
                warmup_steps=warmup_steps,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # -------- 前向传播 --------
            # 文本编码：用 clip 自带 tokenizer，context_length=扩展后的 256
            text_tokens = clip.tokenize(
                texts,
                context_length=model.context_length,
                truncate=True,
            ).to(device)

            # 得到特征
            image_features = model.encode_image(images)     # [B, D]
            text_features = model.encode_text(text_tokens)  # [B, D]

            # 归一化（余弦相似度）
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # CLIP 的 logit_scale
            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()  # [B, B]
            logits_per_text = logits_per_image.t()                               # [B, B]

            # 目标：对角线是正样本
            targets = torch.arange(images.size(0), device=device)

            loss_i = F.cross_entropy(logits_per_image, targets)
            loss_t = F.cross_entropy(logits_per_text, targets)
            loss = (loss_i + loss_t) / 2.0
            batch_losses.append(loss.item())
            # -------- 反向传播 --------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 限制 logit_scale 的范围（CLIP 训练里的做法，防止数值爆炸）
            with torch.no_grad():
                model.logit_scale.clamp_(0, math.log(100))

            pbar.set_postfix(
                loss=loss.item(),
                lr=lr,
            )

            if global_step >= total_steps:
                break
            losses.append(sum(batch_losses) / len(batch_losses))
        if global_step >= total_steps:
            break

    print("Training finished.")
    ## 绘制 loss 曲线
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(losses)
    plt.savefig("loss_vitl14.png")
    # 6) 保存模型（只保存 state_dict 就好）
    save_path = "single_image_clip_vitl14.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved fine-tuned model to {save_path}")


if __name__ == "__main__":
    # 你可以根据自己的需求改这些参数
    train_single_image(
        json_path="data/captioning_dataset_5imgs_train.json",  # 如果你已经拆了 train/val，这里用 train 的
        image_root="data/resized",
        batch_size=32,          # ViT-L/14 比较吃显存，显存不够就调小
        num_epochs=20,           # 先跑通；真正实验时你自己调
        base_lr=1e-5,
        warmup_steps=20000,     # 论文里写的固定 20k
        max_text_len=256,
        text_source="article",  # 或 "caption" / "article+caption"
    )
