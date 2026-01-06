import os
from utils import load_json, save_json, count_json_image_nums, parse_image_dir
from collections import defaultdict

JSON_PATH = os.path.join("data", "captioning_dataset.json")
IMAGE_DIR = os.path.join("data", "resized")
K_IMAGE = [4]
print("_".join([str(item) for item in K_IMAGE]))
OUT_JSON_PATH = os.path.join("data", f"captioning_dataset_{'_'.join([str(item) for item in K_IMAGE])}imgs.json")

def main():
    # 1. 读取 JSON
    data = load_json(JSON_PATH)
    print(f"[JSON] 原始 key 总数: {len(data)}")

    # 2. 扫描图片目录
    _, image_keys, indices_per_key = parse_image_dir(IMAGE_DIR)

    # 3. 统计 JSON 中每个 key 的图片数量
    json_images_per_key = count_json_image_nums(data)

    # 4. 过滤逻辑：
    #    条件 A：JSON 中 images 数量 in K_IMAGE（原为 ==5）
    #    条件 B：key 在图片目录中存在
    #    条件 C：JSON 中这 n 个 index 在图片目录下都存在
    filtered_data = {}
    dropped_because_not_k = 0
    dropped_because_no_img = 0
    dropped_because_incomplete_imgs = 0

    for k, record in data.items():
        img_num_json = json_images_per_key.get(k, 0)

        # 条件 A：JSON 中 images 数量必须为 K
        if img_num_json not in K_IMAGE:
            dropped_because_not_k += 1
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
            dropped_because_not_k += 1
            continue

        # 再次保证长度在 K_IMAGE 中
        if len(expected_indices) not in K_IMAGE:
            dropped_because_not_k += 1
            continue

        # 条件 B：图片目录中必须存在该 key
        if k not in image_keys:
            dropped_because_no_img += 1
            continue

        # 条件 C：目录中该 key 下的 index 集合必须覆盖 JSON 里的 index
        dir_indices = indices_per_key.get(k, set())
        if not expected_indices.issubset(dir_indices):
            dropped_because_incomplete_imgs += 1
            continue

        # 所有条件都满足，保留
        filtered_data[k] = record

    print(f"[FILTER] 原始样本数: {len(data)}")
    print(f"[FILTER] JSON 中 images 数量不在 {K_IMAGE} 被过滤的样本数: {dropped_because_not_k}")
    print(f"[FILTER] 图片目录中不存在对应 key 被过滤的样本数: {dropped_because_no_img}")
    print(f"[FILTER] 目录中缺少某些 JSON 图像 index 被过滤的样本数: {dropped_because_incomplete_imgs}")
    print(f"[FILTER] 最终保留的样本数: {len(filtered_data)}")

    # 5. 导出新的 JSON 文件
    save_json(OUT_JSON_PATH, filtered_data)


if __name__ == "__main__":
    main()