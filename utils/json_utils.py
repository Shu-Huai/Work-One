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