import os, time, argparse
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from model import UNet
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_channels', type=int, default=1, help='图像通道数，1=灰度，3=RGB')
    parser.add_argument('--size', type=int, default=28, help='图像高宽，默认28')
    parser.add_argument('--ckpt', type=str, required=True, help='模型权重路径')
    parser.add_argument('--batch_size', type=int, default=10000, help='总采样图像数')
    parser.add_argument('--max_batch', type=int, default=64, help='单次采样最大批量（避免OOM）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认42）')
    args = parser.parse_args()

    set_seed(args.seed)

    C, H, W = args.image_channels, args.size, args.size
    total = args.batch_size
    max_bs = args.max_batch
    T = 1000

    os.makedirs('samples2', exist_ok=True)
    os.makedirs('fid2', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    perf_log = open('logs/performance_log2_pytorch.txt', 'a')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DDPM参数
    betas = torch.linspace(1e-4, 0.02, T, dtype=torch.float32, device=device)
    alphas = 1 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)
    sqrt_betas = torch.sqrt(betas)

    # 加载模型
    model = UNet(image_channels=C)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device).eval()

    all_samples = []
    start_time = time.time()
    num_batches = (total + max_bs - 1) // max_bs

    print(f"[INFO] 开始采样，共 {total} 张图，分为 {num_batches} 批...")

    with torch.no_grad():
        pbar = tqdm(total=total, desc="Sampling", ncols=80)
        for batch_id in range(num_batches):
            cur_bs = min(max_bs, total - batch_id * max_bs)
            x = torch.randn(cur_bs, C, H, W, device=device)

            for t in reversed(range(T)):
                t_tensor = torch.full((cur_bs,), t, dtype=torch.long, device=device)
                pred_noise = model(x, t_tensor)
                coef = (1 - alphas[t]) / (sqrt_one_minus_alpha_cumprod[t] + 1e-8)
                mean = (x - coef * pred_noise) * sqrt_recip_alphas[t]
                x = mean + sqrt_betas[t] * torch.randn_like(x) if t > 0 else mean

                if batch_id == 0 and (t == T - 1 or t % 100 == 0 or t == 0):
                    grid = to_grid(x, C, H, W)
                    grid.save(f'samples2/step_t{t:04d}.png')

            x_np = ((x + 1) / 2 * 255.0).clamp(0, 255).cpu().numpy().astype(np.uint8)
            for i in range(cur_bs):
                arr = x_np[i]
                img = Image.fromarray(np.transpose(arr, (1, 2, 0)), mode='RGB') if C == 3 else Image.fromarray(arr[0], mode='L')
                img.save(f'fid2/sample_{batch_id * max_bs + i}.png')

            if batch_id == 0:
                all_samples.extend(list(x_np[:64]))
            elif len(all_samples) < 64:
                remain = 64 - len(all_samples)
                all_samples.extend(list(x_np[:remain]))

            pbar.update(cur_bs)
        pbar.close()

    print("正在生成样本网格图 (8x8)...")
    grid_img = Image.new('RGB' if C == 3 else 'L', (W * 8, H * 8))
    for idx, arr in enumerate(all_samples[:64]):
        r, c = divmod(idx, 8)
        tile = Image.fromarray(np.transpose(arr, (1, 2, 0)), mode='RGB') if C == 3 else Image.fromarray(arr[0], mode='L')
        grid_img.paste(tile, (c * W, r * H))
    grid_img.save('samples2/sample_grid.png')

    total_time = time.time() - start_time
    perf_log.write(f"[Sample] Total: {total_time:.2f}s, Avg per image: {total_time/total:.4f}s\n")
    perf_log.close()

    print(f"采样完成，共生成 {total} 张图，保存至 fid/，网格图保存至 samples/sample_grid.png")

def to_grid(x, C, H, W):
    x_np = ((x + 1) / 2 * 255.0).clamp(0, 255).cpu().numpy().astype(np.uint8)
    grid = Image.new('RGB' if C == 3 else 'L', (W * 8, H * 8))
    for i in range(min(64, x.shape[0])):
        r, c = divmod(i, 8)
        arr = x_np[i]
        img = Image.fromarray(np.transpose(arr, (1, 2, 0)), mode='RGB') if C == 3 else Image.fromarray(arr[0], mode='L')
        grid.paste(img, (c * W, r * H))
    return grid

if __name__ == '__main__':
    main()
