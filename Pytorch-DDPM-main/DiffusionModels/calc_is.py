import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset

class ImageFolderDataset(Dataset):
    def __init__(self, folder, image_size=299):
        self.paths = sorted([
            os.path.join(folder, fname)
            for fname in os.listdir(folder)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)

def calculate_inception_score(images, splits=10):
    N = len(images)
    assert N > 0

    preds = images
    scores = []

    for i in range(splits):
        part = preds[i * N // splits: (i + 1) * N // splits]
        py = part.mean(0)
        kl = part * (torch.log(part + 1e-6) - torch.log(py + 1e-6))
        kl = kl.sum(1)
        scores.append(kl.mean().item())

    mean_score = np.exp(np.mean(scores))
    std_score = np.exp(np.std(scores))
    return mean_score, std_score

@torch.no_grad()
def get_inception_preds(dataloader, device):
    model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    preds = []
    for batch in tqdm(dataloader, desc="Extracting features"):
        batch = batch.to(device)
        if batch.shape[2] != 299 or batch.shape[3] != 299:
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
        logits = model(batch)
        preds.append(F.softmax(logits, dim=1).cpu())
    return torch.cat(preds, dim=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='生成图像所在文件夹')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--splits', type=int, default=10, help='划分数（计算 std 用）')
    parser.add_argument('--out', type=str, default='inception_score.txt', help='结果保存路径')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ImageFolderDataset(args.folder)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"[INFO] 图像数量: {len(dataset)}, 使用设备: {device}")
    preds = get_inception_preds(dataloader, device)
    mean, std = calculate_inception_score(preds, splits=args.splits)

    result = f"Inception Score: {mean:.4f} ± {std:.4f}"
    print("[✓] " + result)
    with open(args.out, 'w') as f:
        f.write(result + "\n")

if __name__ == '__main__':
    main()
