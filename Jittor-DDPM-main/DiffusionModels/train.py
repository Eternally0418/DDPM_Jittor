import os, time, argparse
import numpy as np
import matplotlib.pyplot as plt
import jittor as jt
from jittor import nn
from model import UNet

jt.flags.use_cuda = 1

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)        #把种子set到全局

def load_saved_dataset(path, batch_size):
    data = np.load(path)        #加载 .npz 文件
    images, labels = data['images'], data['labels']     #从其中取出键为 'images' 和 'labels' 的数组
    indices = np.random.permutation(len(images))
    images, labels = images[indices], labels[indices] #打乱样本顺序
    for i in range(0, len(images), batch_size):
        yield jt.array(images[i:i+batch_size]), jt.array(labels[i:i+batch_size])
            #循环从打乱后的 images 和 labels 中切出一个个 batch，使用 jt.array() 转换成 Jittor 张量，
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--image_channels', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    dataset_name = args.dataset
    data_path = f"data/{dataset_name}_processed.npz"
    image_channels = args.image_channels if args.image_channels is not None else (1 if dataset_name == "mnist" else 3)

    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)       #如果不存在 logs/ 和 checkpoints/ 文件夹，就创建它们
    perf_log = open('logs/performance_log2_jittor.txt', 'w')
    train_log = open('logs/train_log_jittor2.txt', 'w')     #'w' 模式意味着会覆盖旧文件，如需追加应使用 'a'

    model = UNet(image_channels=image_channels)
    optimizer = nn.Adam(model.parameters(), lr=args.lr)

    T = args.T
    betas = np.linspace(1e-4, 0.02, T, dtype=np.float32)        #定义一个线性变化的 β 序列：β₁, β₂, ..., β_T
    alphas = 1 - betas
    alpha_cumprod = np.cumprod(alphas)  #累积乘积 ᾱₜ
    sqrt_alpha_cumprod = np.sqrt(alpha_cumprod)     #	原图缩放因子
    sqrt_one_minus_alpha_cumprod = np.sqrt(1 - alpha_cumprod)       #	噪声缩放因子

    loss_history = []

    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_loader = load_saved_dataset(data_path, args.batch_size)

        for batch_idx, (x0, _) in enumerate(train_loader):
            if x0.ndim == 3:        #检查 x0 的维度：如果是 [B, H, W]（即没有通道维度）
                x0 = x0.unsqueeze(1)        #则用 unsqueeze(1) 插入一个通道维度，变成 [B, 1, H, W]
            B = x0.shape[0]
            t = np.random.randint(0, T, size=B, dtype=np.int32) #从整数区间 [0, T) 中随机生成 B 个整数，作为时间步索引
            t_var = jt.array(t)     #将 NumPy 数组 t 转换为 Jittor 框架的张量 t_var

            noise = jt.randn(x0.shape)      #生成与 x0 相同形状的标准正态分布噪声
            sqrt_alpha_t = jt.array(sqrt_alpha_cumprod[t]).reshape((B,1,1,1))   #sqrt_alpha_cumprod[t] 是 shape 为 [B] 的数组（每个样本对应一个时间步的缩放系数 √ᾱ_t）;reshape 成 [B, 1, 1, 1] 以便在 x0 的每个像素上进行广播乘法
            sqrt_one_minus_t = jt.array(sqrt_one_minus_alpha_cumprod[t]).reshape((B,1,1,1))     #这是每个样本时间步的 √(1 - ᾱ_t)，同样 reshape 成 [B, 1, 1, 1]
            x_t = sqrt_alpha_t * x0 + sqrt_one_minus_t * noise  #前向扩散公式由x0->xt

            pred_noise = model(x_t, t_var)  #将加噪样本 x_t 和时间步 t_var 作为输入送入模型;预测出当前步的噪声 ε（也就是之前加进去的 noise）
            loss = ((pred_noise - noise) ** 2).mean()   # 均方误差（MSE）损失函数
            optimizer.step(loss)        #用优化器（如 Adam）对模型参数执行一次反向传播并更新参数

            loss_val = float(loss)  #将 loss（一个 Jittor 张量）转换成 Python 标量 float
            train_log.write(f"{loss_val:.6f}\n")        #将 loss 值写入日志文件 train_log，保留 6 位小数
            loss_history.append(loss_val)       #用来画loss曲线的

            if batch_idx % 100 == 0:
                print(f"[Epoch {epoch+1} Batch {batch_idx}] Loss: {loss_val:.6f}")

        epoch_time = time.time() - epoch_start
        perf_log.write(f"Epoch {epoch+1} Time: {epoch_time:.2f} sec\n")

        model.save(f"checkpoints/ddpm_{dataset_name}_epoch{epoch+1}.ckpt")

        plt.figure()
        plt.plot(loss_history)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.savefig(f"logs/loss_curve_{dataset_name}.png")
        plt.close()

        perf_log.flush()
        train_log.flush()

    model.save(f"ddpm_{dataset_name}.ckpt")
    print("训练完成")
    perf_log.close()
    train_log.close()

if __name__ == '__main__':
    main()