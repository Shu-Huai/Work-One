#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from collections import defaultdict

JSON_PATH = os.path.join("data", "captioning_dataset.json")
IMAGE_DIR = os.path.join("data", "resized")
OUT_JSON_PATH = os.path.join("data", "captioning_dataset_5imgs.json")


def load_json(path):
    print(f"加载 JSON: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("顶层不是 object（字典），脚本假设顶层是 {id: record} 结构")
    return data


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
        full_path = os.path.join(image_dir, fname)
        if not os.path.isfile(full_path):
            continue

        lower = fname.lower()
        if not lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            continue

        name, _ext = os.path.splitext(fname)

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


def main():
    # 1. 读取 JSON
    data = load_json(JSON_PATH)
    print(f"[JSON] 原始 key 总数: {len(data)}")

    # 2. 扫描图片目录
    images_per_key_dir, image_keys, indices_per_key = parse_image_dir(IMAGE_DIR)

    # 3. 统计 JSON 中每个 key 的图片数量
    json_images_per_key = count_json_image_nums(data)

    # 4. 过滤逻辑：
    #    条件 A：JSON 中 images 数量 == 5
    #    条件 B：key 在图片目录中存在
    #    条件 C：JSON 中这 5 个 index 在图片目录下都存在（不能缺 _2 这种情况）
    filtered_data = {}
    dropped_because_not5 = 0
    dropped_because_no_img = 0
    dropped_because_incomplete_imgs = 0

    for k, record in data.items():
        img_num_json = json_images_per_key.get(k, 0)

        # 条件 A：JSON 中 images 数量必须为 5
        if img_num_json != 5:
            dropped_because_not5 += 1
            continue

        # 从 JSON 中解析“期望的 index 集合”
        images_field = record.get("images")
        if isinstance(images_field, dict):
            expected_indices = set(images_field.keys())
        elif isinstance(images_field, list):
            # 如果是 list，默认认为文件名是 key_0.jpg, key_1.jpg, ...
            expected_indices = {str(i) for i in range(len(images_field))}
        else:
            # 其他类型，视为不合法
            dropped_because_not5 += 1
            continue

        # 再次保证长度为 5（防御性检查）
        if len(expected_indices) != 5:
            dropped_because_not5 += 1
            continue

        # 条件 B：图片目录中必须存在该 key
        if k not in image_keys:
            dropped_because_no_img += 1
            continue

        # 条件 C：目录中该 key 下的 index 集合必须覆盖 JSON 里的 5 个 index
        dir_indices = indices_per_key.get(k, set())
        # 例如 JSON 有 {0,1,2,3,4}，但目录只有 {0,1,3,4}，则不通过
        if not expected_indices.issubset(dir_indices):
            dropped_because_incomplete_imgs += 1
            # 调试时可以开这一行，看看缺了哪些：
            # print(f"[MISS] key={k}, JSON indices={expected_indices}, DIR indices={dir_indices}")
            continue

        # 所有条件都满足，保留
        filtered_data[k] = record

    print(f"[FILTER] 原始样本数: {len(data)}")
    print(f"[FILTER] JSON 中 images 数量 != 5 被过滤的样本数: {dropped_because_not5}")
    print(f"[FILTER] 图片目录中不存在对应 key 被过滤的样本数: {dropped_because_no_img}")
    print(f"[FILTER] 目录中缺少某些 JSON 图像 index 被过滤的样本数: {dropped_because_incomplete_imgs}")
    print(f"[FILTER] 最终保留的样本数: {len(filtered_data)}")

    # 5. 导出新的 JSON 文件
    out_dir = os.path.dirname(OUT_JSON_PATH)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] 已将过滤后的数据保存到: {OUT_JSON_PATH}")


if __name__ == "__main__":
    main()
