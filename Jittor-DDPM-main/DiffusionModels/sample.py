import os, time, argparse
import numpy as np
from PIL import Image
import jittor as jt
from model import UNet
from tqdm import tqdm
import random

jt.flags.use_cuda = 1

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_channels', type=int, default=1, help='图像通道数，1=灰度，3=RGB')
    parser.add_argument('--size', type=int, default=28, help='图像高宽，默认28')
    parser.add_argument('--ckpt', type=str, required=True, help='模型权重路径')
    parser.add_argument('--batch_size', type=int, default=10000, help='总采样图像数')
    parser.add_argument('--max_batch', type=int, default=64, help='单次最大采样数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认42）')
    args = parser.parse_args()

    set_seed(args.seed)

    C, H, W = args.image_channels, args.size, args.size
    #图像维度将为 [B, C, H, W]；通道数 C，高度 H 和宽度 W
    total = args.batch_size
    max_bs = args.max_batch
    T = 1000

    os.makedirs('samples2', exist_ok=True)
    os.makedirs('fid2', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    perf_log = open('../../Pytorch-DDPM-main/DiffusionModels/logs/performance_log2_jittor.txt', 'a')

    # DDPM调度表
    betas = np.linspace(1e-4, 0.02, T, dtype=np.float32)
    alphas = 1 - betas
    alpha_cumprod = np.cumprod(alphas)
    sqrt_recip_alphas = 1.0 / np.sqrt(alphas)
    sqrt_one_minus_alpha_cumprod = np.sqrt(1 - alpha_cumprod)
    sqrt_betas = np.sqrt(betas)

    # 加载模型
    model = UNet(image_channels=C)
    model.load(args.ckpt)
    model.eval()        #设置模型为评估模式关闭 Dropout关闭 BatchNorm 的统计更新



    all_samples = []
    start_time = time.time()
    num_batches = (total + max_bs - 1) // max_bs        #将总采样量 total 划分成若干个批次，每个最多 max_bs 个样本

    print(f"开始采样，共 {total} 张图，分为 {num_batches} 批...")
    with tqdm(total=total, desc="Sampling", ncols=80) as pbar:
        for batch_id in range(num_batches):
            cur_bs = min(max_bs, total - batch_id * max_bs)
            x = jt.randn((cur_bs, C, H, W))     #	[cur_bs, C, H, W]   	批量噪声图像（标准正态分布）

            for t in reversed(range(T)):
                t_array = jt.full((cur_bs,), t, dtype='int32')  #为当前 batch 创建一个形如 [t, t, ..., t] 的数组，长度为 cur_bs
                pred_noise = model(x, t_array)
                coef = (1 - alphas[t]) / (sqrt_one_minus_alpha_cumprod[t] + 1e-8)
                mean = (x - coef * pred_noise) * sqrt_recip_alphas[t]
                x = mean + sqrt_betas[t] * jt.randn(x.shape) if t > 0 else mean

                if batch_id == 0 and (t == T - 1 or t % 100 == 0 or t == 0):
                    to_grid(x, C, H, W).save(f'samples2/step_t{t:04d}.png')     #只对第一个 batch 可视化 网格展示

            x_np = ((x.data + 1) / 2 * 255.0).clip(0, 255).astype(np.uint8) #反归一化处理最终 x_np 变为形状为 [B, C, H, W]，类型为 uint8 的图像数据（0~255）
            for i in range(cur_bs):
                arr = x_np[i]
                img = Image.fromarray(np.transpose(arr, (1, 2, 0)), mode='RGB') if C == 3 else Image.fromarray(arr[0], mode='L')
                img.save(f'fid2/sample_{batch_id * max_bs + i}.png')        #使用 np.transpose(..., (1, 2, 0)) 将图像从 [C, H, W] 转为 [H, W, C]  或者取第 0 通道 arr[0] 作为 [H, W]

            if batch_id == 0:       #网格图保存
                all_samples.extend(list(x_np[:64]))
            elif len(all_samples) < 64:
                all_samples.extend(list(x_np[:64 - len(all_samples)]))

            pbar.update(cur_bs)

    print("正在保存网格图 sample_grid.png ...")
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
    x_np = ((x.data + 1) / 2 * 255.0).clip(0, 255).astype(np.uint8)
    grid = Image.new('RGB' if C == 3 else 'L', (W * 8, H * 8))
    for i in range(min(64, x.shape[0])):
        r, c = divmod(i, 8)
        arr = x_np[i]
        img = Image.fromarray(np.transpose(arr, (1, 2, 0)), mode='RGB') if C == 3 else Image.fromarray(arr[0], mode='L')
        grid.paste(img, (c * W, r * H))
    return grid

if __name__ == '__main__':
    main()
