import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataset import CaptioningDataset, captioning_collate_fn
import matplotlib.pyplot as plt
import time
import numpy as np


if __name__ == "__main__":
    # 你给定的路径
    json_path = "data/captioning_dataset.json"
    image_root = "data/resized"

    # 图像已经 resize 成 256x256，这里只做 ToTensor（如有需要可加 Normalize）
    transform = transforms.ToTensor()

    dataset = CaptioningDataset(
        json_path=json_path,
        image_root=image_root,
        transform=transform,
        use_headline=False,  # 如果想把 headline 也拼进 article 就改成 True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,         # 根据你机器情况调整
        collate_fn=captioning_collate_fn,
    )

for batch in dataloader:
    articles = batch["articles"]            # List[str]，len = B
    imgs_per_article = batch["images"]      # List[Tensor]，每个 [Ni, 3, 256, 256]
    captions_per_article = batch["captions"]  # List[List[str]]

    # 举个例子：取第 0 条样本
    article0 = articles[0]
    images0 = imgs_per_article[0]           # shape [N0, 3, 256, 256]
    captions0 = captions_per_article[0]     # len = N0
    if len(captions0)==1:
        continue
    # ---------------------- 新增：显示图片+打印文本 ----------------------
    # 1. 打印完整文本信息
    print("="*50)
    print(f"【文章完整文本】\n{article0}\n")
    print(f"【共 {len(captions0)} 张图片，对应描述如下】")
    for idx, cap in enumerate(captions0):
        print(f"图片{idx+1}：{cap}")
    print("="*50)

    # 2. 显示所有图片（自动排版，适配任意数量）
    N = len(images0)  # 当前样本的图片总数
    rows = (N + 2) // 3  # 每行3张图，向上取整（如4张图→2行）
    cols = min(N, 3)     # 每行最多3张图

    plt.figure(figsize=(15, 5*rows))  # 调整画布大小（适配多行）
    # plt.suptitle("当前样本的图片及对应描述", fontsize=16, y=0.95)  # 总标题

    for i in range(N):
        # Tensor转numpy：[3, H, W] → [H, W, 3]，并还原像素值（默认Tensor是0-1或0-255）
        img = images0[i].cpu().detach().numpy().transpose(1, 2, 0)
        # 若Tensor像素值是0-1，转成0-255（根据你的数据格式选择，二选一）
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)

        # 子图布局
        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(img)
        # ax.set_title(f"图片{i+1}：{captions0[i]}", fontsize=10, wrap=True)  # 图片标题=描述
        ax.axis("off")  # 隐藏坐标轴

    plt.tight_layout()  # 自动调整子图间距
    plt.savefig("aaa.png")

    # 3. 休眠30秒（期间图片窗口保持打开）
    time.sleep(30)

    # 4. 关闭图片窗口，释放内存（避免累积）
    plt.close("all")

    # ---------------------- 原有逻辑 ----------------------
    # 在这里喂给模型即可……
    # break