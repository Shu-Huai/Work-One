#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")  # 无图形界面也能画图
import matplotlib.pyplot as plt

JSON_PATH = os.path.join("data", "captioning_dataset_5imgs.json")
IMAGE_DIR = os.path.join("data", "resized")


def load_json(path):
    print(f"加载 JSON: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("顶层不是 object（字典），脚本假设顶层是 {id: record} 结构")
    return data

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


def parse_image_dir(image_dir):
    """
    扫描图片目录，解析出：
    - images_per_key: key -> 该 key 下有多少张图片
    - image_keys: 出现在图片目录中的所有 key 集合
    """
    print(f"扫描图片目录: {image_dir}")
    images_per_key = defaultdict(int)

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")

    for fname in os.listdir(image_dir):
        full_path = os.path.join(image_dir, fname)
        if not os.path.isfile(full_path):
            continue

        # 只考虑常见图片扩展名
        lower = fname.lower()
        if not lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            continue

        name, _ext = os.path.splitext(fname)

        # 约定命名: <key>_<编号>，例如 4fd2a7be8eb7c8105d89098d_1.jpg
        try:
            key_part, idx_part = name.rsplit("_", 1)
        except ValueError:
            # 不是我们期望的格式，跳过或根据需要打印警告
            # print(f"跳过非标准文件名: {fname}")
            continue

        # 可以根据需要验证 idx_part 是否为数字，这里不强制
        images_per_key[key_part] += 1

    image_keys = set(images_per_key.keys())
    return images_per_key, image_keys


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
            # 其他类型不常见，保守起见按 0 处理
            n = 0
        json_images_per_key[k] = n
    return json_images_per_key


def main():
    # 1. 读取 JSON
    data = load_json(JSON_PATH)
    # 1.5 检查每个 key 的 value 是否满足既定格式
    invalid_entries = {}
    for k, record in data.items():
        if not is_valid_record(record):
            invalid_entries[k] = record

    invalid_count = len(invalid_entries)
    print(f"[CHECK] 不满足指定格式的 key 数量: {invalid_count}")

    if invalid_entries:
        error_path = "error.json"
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(invalid_entries, f, ensure_ascii=False, indent=2)
        print(f"[CHECK] 已将不满足格式的条目保存到: {error_path}")
    # 2. 统计 JSON 中 key 的个数
    json_keys = set(data.keys())
    json_key_count = len(json_keys)
    print(f"[JSON] key 的个数: {json_key_count}")

    # 3 & 4. 扫描图片目录，统计图片目录中的 key 个数，并比较
    images_per_key_dir, image_keys = parse_image_dir(IMAGE_DIR)
    image_key_count = len(image_keys)
    print(f"[IMG ] 图片目录中的 key 个数: {image_key_count}")

    same_key_count = (json_key_count == image_key_count)
    print(f"JSON key 个数 与 图片目录 key 个数是否相等: {same_key_count}")

    # 额外：看看谁多谁少（可选）
    missing_in_imgs = json_keys - image_keys
    missing_in_json = image_keys - json_keys
    print(f"仅在 JSON 中存在但没有图片的 key 数: {len(missing_in_imgs)}")
    print(f"仅在图片目录中存在但 JSON 没有的 key 数: {len(missing_in_json)}")

    # 5. 统计「一个 key 对应多个图片」的条目个数（JSON vs 图片目录）并比较
    json_images_per_key = count_json_image_nums(data)

    json_multi_img_count = sum(1 for k, n in json_images_per_key.items() if n > 1)
    dir_multi_img_count = sum(1 for k, n in images_per_key_dir.items() if n > 1)

    print(f"[JSON] 一个 key 对应多张图片的条目个数: {json_multi_img_count}")
    print(f"[IMG ] 一个 key 对应多张图片的条目个数: {dir_multi_img_count}")
    same_multi = (json_multi_img_count == dir_multi_img_count)
    print(f"JSON vs 图片目录 多图条目个数是否相等: {same_multi}")

    # 6. 使用 matplotlib 统计 JSON 中每个 key 对应的图片数量分布，并画柱状图
    #    横轴: 每个 key 拥有的图片数量 (1张, 2张, 3张, ...)
    #    纵轴: 拥有该数量图片的 key 个数

    counter = Counter()
    for k, n in json_images_per_key.items():
        counter[n] += 1

    # 一般我们只画 n > 0 的情况，0 张图的可以忽略，也可以视需求保留
    nums = sorted(n for n in counter.keys() if n > 0)
    counts = [counter[n] for n in nums]

    print("JSON 中图片数量分布（只统计 n>0）：")
    for n in nums:
        print(f"  {n} 张图的 key 个数: {counter[n]}")

    if not nums:
        print("没有任何 key 拥有图片（n>0），跳过画图。")
        return

    plt.figure(figsize=(8, 5))
    plt.bar([str(n) for n in nums], counts)
    plt.xlabel("Images per key")
    plt.ylabel("Counts")
    plt.title("Distribution of images per key")
    plt.tight_layout()

    out_png = "image_count_hist.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"直方图已保存为: {out_png}")


if __name__ == "__main__":
    main()
