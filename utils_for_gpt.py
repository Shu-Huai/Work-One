# utils and metrics package for reference
import torch
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


from collections import defaultdict
import os


def parse_image_dir(image_dir):
    """
    扫描图片目录，解析出：
    - images_per_key: key -> 该 key 下有多少张图片
    - image_keys: 出现在图片目录中的所有 key 集合
    - indices_per_key: key -> 该 key 下出现过的“编号字符串”集合
      例如:  56cd7f..._0.jpg, 56cd7f..._1.jpg
      则 indices_per_key["56cd7f..."] = {"0", "1"}
    """
    print(f"扫描图片目录: {image_dir}")
    images_per_key = defaultdict(int)
    indices_per_key = defaultdict(set)

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")

    for fname in os.listdir(image_dir):
        # 跳过子目录、符号链接、设备文件等
        full_path = os.path.join(image_dir, fname)
        if not os.path.isfile(full_path):
            continue
        
        # 跳过非图片
        lower = fname.lower()
        if not lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            continue

        name, _ = os.path.splitext(fname)

        try:
            key_part, idx_part = name.rsplit("_", 1)
        except ValueError:
            # 非标准命名，忽略
            # print(f"跳过非标准文件名: {fname}")
            continue

        images_per_key[key_part] += 1
        indices_per_key[key_part].add(idx_part)

    image_keys = set(images_per_key.keys())
    print(f"[IMG ] 图片目录中的 key 个数: {len(image_keys)}")
    return images_per_key, image_keys, indices_per_key

import json
import os

def load_json(path):
    print(f"加载 JSON: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("顶层不是 object（字典），脚本假设顶层是 {id: record} 结构")
    return data


def save_json(path, data):
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] 保存 JSON: {path}，样本数: {len(data)}")


def count_json_image_nums(data):
    """
    统计 JSON 中每个 key 拥有的图片数量：
    返回:
    - json_images_per_key: key -> 图片数量
    """
    json_images_per_key = {}
    for k, record in data.items():
        images_field = record.get("images")
        if isinstance(images_field, dict):
            n = len(images_field)
        elif isinstance(images_field, list):
            n = len(images_field)
        elif images_field is None:
            n = 0
        else:
            n = 0
        json_images_per_key[k] = n
    return json_images_per_key


def is_valid_record(record):
    """
    检查一条记录是否满足如下格式：
    {
      "images": { ... },          # 必须是 dict
      "headline": { ... },        # 必须是 dict
      "abstract": null 或 str,
      "article_url": str,
      "article": str
    }
    """
    if not isinstance(record, dict):
        return False

    required_keys = ["images", "headline", "abstract", "article_url", "article"]
    for k in required_keys:
        if k not in record:
            return False

    if not isinstance(record["images"], dict):
        return False

    if not isinstance(record["headline"], dict):
        return False

    abstract = record["abstract"]
    if abstract is not None and not isinstance(abstract, str):
        return False

    if not isinstance(record["article_url"], str):
        return False

    if not isinstance(record["article"], str):
        return False

    return True


import random
import os
import numpy as np
import torch

def set_all_seeds(seed: int = 42):
    # Python 内置 random
    random.seed(seed)
    
    # Python 环境变量（影响某些库的行为）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU
    
    # 为了进一步保证可复现性（但会降低性能）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


import math
def cosine_with_warmup_lr(step: int, total_steps: int, warmup_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    warmup_steps = min(warmup_steps, total_steps)
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def cast_trainable_params_to_fp32(model):
    """
    避免 GradScaler 在 fp16 参数上 unscale 报错：把 requires_grad 的参数转 fp32
    """
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()


import tqdm
import torch
from typing import Dict
@torch.no_grad()
def retrieval_metrics_multi(
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
    N, _ = text_embs_cpu.shape
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
