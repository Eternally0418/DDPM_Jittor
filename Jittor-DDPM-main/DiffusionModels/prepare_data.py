import os
import numpy as np
import jittor as jt
from jittor.dataset.mnist import MNIST
from jittor.dataset.cifar import CIFAR10
import jittor.transform as transform
from PIL import Image

def prepare_builtin(dataset_name, save_path):
    if dataset_name == 'mnist':
        dataset = MNIST(train=True, transform=transform.Compose([
            transform.Resize(28),
            transform.Gray(),
            transform.ImageNormalize(mean=[0.5], std=[0.5])     #将像素从 [0,255] 归一化到 [-1,1]
        ])).set_attrs(batch_size=60000, shuffle=False)      #一次性取出所有图像

    elif dataset_name == 'cifar10':
        dataset = CIFAR10(train=True, transform=transform.Compose([
            transform.Resize(32),
            transform.ImageNormalize(mean=[0.5], std=[0.5])
        ])).set_attrs(batch_size=50000, shuffle=False)

    else:
        raise ValueError("Unsupported built-in dataset")

    images, labels = next(iter(dataset))    #iter(dataset) 中只有一个 batch；从迭代器中取出这个 batch
    images = images.numpy()
    labels = labels.numpy()

    if images.ndim == 3:        #判断维度是 3，就手动添加一个通道维度，变为 [N, 1, H, W]
        images = images.reshape((-1, 1, images.shape[1], images.shape[2]))

    print(f"加载 {dataset_name} 成功，图像形状: {images.shape}，标签数量: {labels.shape}")
    np.savez_compressed(save_path, images=images, labels=labels)
    print(f"Built-in {dataset_name} 预处理完成，保存至 {save_path}")

def prepare_custom_image_folder(root_dir, save_path, image_size=32):
    images, labels = [], []
    class_map = {}
    class_idx = 0
    total_images = 0

    for class_name in sorted(os.listdir(root_dir)):
        class_folder = os.path.join(root_dir, class_name)   #拼接成每个类的完整路径
        if not os.path.isdir(class_folder):     #检查是否为文件夹
            continue
        if class_name not in class_map:
            class_map[class_name] = class_idx       #为每个类别分配一个唯一的整数标签 ID，构建一个 类名 → 类别索引 的映射表
            class_idx += 1

        for fname in os.listdir(class_folder):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            img_path = os.path.join(class_folder, fname)        #完整图像路径
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((image_size, image_size))
                img_np = np.asarray(img).astype(np.float32) / 255.0
                img_np = (img_np - 0.5) / 0.5  # Normalize to [-1,1]
                images.append(img_np.transpose(2, 0, 1))  # 图像原始格式为 [H, W, C]，需要转置为 [C, H, W]（PyTorch、Jittor标准）
                labels.append(class_map[class_name])
                total_images += 1
            except Exception as e:
                print(f"跳过无法读取的图像: {img_path}，错误: {e}")

    if total_images == 0:
        raise RuntimeError(f"没有找到任何图像！请检查路径是否正确: {root_dir}")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    print(f"成功处理图像数量: {total_images}")
    print(f"图像 shape: {images.shape}，标签 shape: {labels.shape}")
    print(f"类别映射: {class_map}")

    np.savez_compressed(save_path, images=images, labels=labels)
    print(f"自定义数据集预处理完成，保存至 {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help="mnist | cifar10 | 自定义路径（如 data/my_dataset）")
    parser.add_argument('--out', type=str, default=None, help="保存路径 .npz")
    args = parser.parse_args()

    if args.dataset in ['mnist', 'cifar10']:
        out_path = args.out or f"data/{args.dataset}_processed.npz"
        prepare_builtin(args.dataset, out_path)
    else:
        dataset_path = args.dataset
        name = os.path.basename(os.path.normpath(dataset_path))     #获取文件夹名
        out_path = args.out or f"data/{name}_processed.npz"
        prepare_custom_image_folder(dataset_path, out_path)
