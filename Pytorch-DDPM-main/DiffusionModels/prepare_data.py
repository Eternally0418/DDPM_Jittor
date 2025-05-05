import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms

def prepare_builtin(dataset_name, save_path):
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    else:
        raise ValueError("Unsupported built-in dataset")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    images, labels = next(iter(dataloader))
    images = images.numpy()
    labels = labels.numpy()

    print(f"[INFO] 加载 {dataset_name} 成功，图像形状: {images.shape}，标签数量: {labels.shape}")
    np.savez_compressed(save_path, images=images, labels=labels)
    print(f"[✓] Built-in {dataset_name} 预处理完成，保存至 {save_path}")

def prepare_custom_image_folder(root_dir, save_path, image_size=32):
    images, labels = [], []
    class_map = {}
    class_idx = 0
    total_images = 0

    for class_name in sorted(os.listdir(root_dir)):
        class_folder = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_folder):
            continue
        if class_name not in class_map:
            class_map[class_name] = class_idx
            class_idx += 1

        for fname in os.listdir(class_folder):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            img_path = os.path.join(class_folder, fname)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((image_size, image_size))
                img_np = np.asarray(img).astype(np.float32) / 255.0
                img_np = (img_np - 0.5) / 0.5  # Normalize to [-1,1]
                images.append(img_np.transpose(2, 0, 1))  # CHW
                labels.append(class_map[class_name])
                total_images += 1
            except Exception as e:
                print(f"[WARN] 跳过无法读取的图像: {img_path}，错误: {e}")

    if total_images == 0:
        raise RuntimeError(f"[ERROR] 没有找到任何图像！请检查路径是否正确: {root_dir}")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    print(f"[INFO] 成功处理图像数量: {total_images}")
    print(f"[INFO] 图像 shape: {images.shape}，标签 shape: {labels.shape}")
    print(f"[INFO] 类别映射: {class_map}")

    np.savez_compressed(save_path, images=images, labels=labels)
    print(f"[✓] 自定义数据集预处理完成，保存至 {save_path}")

if __name__ == "__main__":
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
        name = os.path.basename(os.path.normpath(dataset_path))
        out_path = args.out or f"data/{name}_processed.npz"
        prepare_custom_image_folder(dataset_path, out_path)
