import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

def load_real_images(npz_path):
    data = np.load(npz_path)
    images = torch.tensor(data["images"], dtype=torch.float32)  # (N, 1, 28, 28)
    images = (images + 1.0) / 2.0  # [-1,1] → [0,1]
    return images

def load_fake_image_paths(folder):
    return sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(".png")
    ])

def preprocess_real_batch(batch, device):
    batch = batch.repeat(1, 3, 1, 1)  # [B, 1, 28, 28] → [B, 3, 28, 28]
    batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
    return (batch * 255).round().to(torch.uint8).to(device)

def preprocess_fake_batch(paths, device):
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((299, 299), Image.BILINEAR)
        img_tensor = transforms.ToTensor()(img) * 255
        imgs.append(img_tensor.to(torch.uint8))
    return torch.stack(imgs).to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', type=str, required=True, help='真实图像 .npz 文件路径')
    parser.add_argument('--fake', type=str, required=True, help='生成图像文件夹路径')
    parser.add_argument('--out', type=str, default="fid_score.txt", help='结果输出日志路径')
    parser.add_argument('--batch_size', type=int, default=64, help='处理批大小，默认64')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    print("[1] 分批处理真实图像...")
    real_images = load_real_images(args.real)
    for i in tqdm(range(0, len(real_images), args.batch_size)):
        batch = preprocess_real_batch(real_images[i:i+args.batch_size], device)
        fid_metric.update(batch, real=True)

    print("[2] 分批处理生成图像...")
    fake_paths = load_fake_image_paths(args.fake)
    for i in tqdm(range(0, len(fake_paths), args.batch_size)):
        batch_paths = fake_paths[i:i+args.batch_size]
        batch = preprocess_fake_batch(batch_paths, device)
        fid_metric.update(batch, real=False)

    print("[3] 计算 FID...")
    score = fid_metric.compute().item()
    print(f"[✓] FID Score: {score:.4f}")
    with open(args.out, 'w') as f:
        f.write(f"FID score: {score:.4f}\n")

if __name__ == '__main__':
    main()
