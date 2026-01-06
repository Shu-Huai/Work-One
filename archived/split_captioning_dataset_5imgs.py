#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random

# ===== 路径配置 =====
# 这里默认你前一个脚本已经生成了这个文件
SRC_JSON_PATH = os.path.join("data", "captioning_dataset_5imgs.json")

# 输出的 train / test 文件
TRAIN_JSON_PATH = os.path.join("data", "captioning_dataset_5imgs_train_60.json")
TEST_JSON_PATH = os.path.join("data", "captioning_dataset_5imgs_test_40.json")

# 训练集比例（70%）
TRAIN_RATIO = 0.6

# 为了可复现，固定随机种子
RANDOM_SEED = 42


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


def main():
    # 1. 加载已经过滤好的 captioning_dataset_5imgs.json
    data = load_json(SRC_JSON_PATH)
    all_keys = list(data.keys())
    total = len(all_keys)
    print(f"[INFO] 总样本数: {total}")

    # 2. 打乱 key，做随机划分
    random.seed(RANDOM_SEED)
    random.shuffle(all_keys)

    # 3. 计算 train / test 数量
    n_train = int(total * TRAIN_RATIO)
    train_keys = all_keys[:n_train]
    test_keys = all_keys[n_train:]

    print(f"[SPLIT] 训练集比例: {TRAIN_RATIO * 100:.1f}%")
    print(f"[SPLIT] 训练集样本数: {len(train_keys)}")
    print(f"[SPLIT] 测试集样本数: {len(test_keys)}")

    # 4. 构建两个字典
    train_data = {k: data[k] for k in train_keys}
    test_data = {k: data[k] for k in test_keys}

    # 5. 保存到文件
    save_json(TRAIN_JSON_PATH, train_data)
    save_json(TEST_JSON_PATH, test_data)


if __name__ == "__main__":
    main()
