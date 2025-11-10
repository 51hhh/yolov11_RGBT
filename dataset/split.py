import os
import random
from pathlib import Path

def create_dataset_split(image_dir, output_dir, train_ratio=0.8):
    """
    扫描指定目录下的PNG图片，按比例分割为训练集和验证集，并生成txt文件。

    Args:
        image_dir (str or Path): 包含PNG图像的目录 (例如 'dataset/images/visible')。
        output_dir (str or Path): 保存 train.txt 和 val.txt 的目录 (例如 'dataset')。
        train_ratio (float): 训练集所占的比例。
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 查找所有 .png 文件
    image_files = sorted(list(image_dir.glob("*.png")))

    if not image_files:
        print(f"警告: 在 '{image_dir}' 目录中没有找到任何 .png 文件。")
        return

    # 随机打乱文件列表以确保分割是随机的
    random.seed(42)  # 使用固定种子保证每次运行结果一致
    random.shuffle(image_files)

    # 计算分割点
    split_index = int(len(image_files) * train_ratio)

    # 分割训练集和验证集
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    # 定义输出文件路径
    train_txt_path = output_dir / "train.txt"
    val_txt_path = output_dir / "val.txt"

    # 写入文件，使用 posix 路径格式 (/)
    with open(train_txt_path, "w") as f:
        for file_path in train_files:
            # 将Windows路径转换为相对的posix路径
            relative_path = file_path.relative_to(output_dir.parent).as_posix()
            f.write(f"{relative_path}\n")

    with open(val_txt_path, "w") as f:
        for file_path in val_files:
            relative_path = file_path.relative_to(output_dir.parent).as_posix()
            f.write(f"{relative_path}\n")

    print(f"数据集分割完成。")
    print(f"总共 {len(image_files)} 个文件。")
    print(f"训练集: {len(train_files)} 个文件，已保存到 '{train_txt_path}'")
    print(f"验证集: {len(val_files)} 个文件，已保存到 '{val_txt_path}'")


if __name__ == "__main__":
    # --- 配置 ---
    # 图片所在的目录
    IMAGE_DIRECTORY = "dataset/images/visible"
    # train.txt 和 val.txt 输出的目录
    OUTPUT_DIRECTORY = "dataset"
    # 训练集比例
    TRAIN_SPLIT_RATIO = 0.8
    # --- 运行 ---
    create_dataset_split(IMAGE_DIRECTORY, OUTPUT_DIRECTORY, TRAIN_SPLIT_RATIO)