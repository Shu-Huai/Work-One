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