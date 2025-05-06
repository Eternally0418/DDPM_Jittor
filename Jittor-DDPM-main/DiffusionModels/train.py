# train_jittor.py
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
    jt.set_global_seed(seed)

def load_saved_dataset(path, batch_size, split='train', val_ratio=0.1):
    data = np.load(path)
    images, labels = data['images'], data['labels']
    indices = np.random.permutation(len(images))
    val_size = int(len(images) * val_ratio)

    if split == 'train':
        indices = indices[val_size:]
    elif split == 'val':
        indices = indices[:val_size]

    images, labels = images[indices], labels[indices]
    for i in range(0, len(images), batch_size):
        yield jt.array(images[i:i+batch_size]), jt.array(labels[i:i+batch_size])

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
    os.makedirs('checkpoints', exist_ok=True)
    perf_log = open('../../Pytorch-DDPM-main/DiffusionModels/logs/performance_log2_jittor.txt', 'w')
    train_log = open('logs/train_log_jittor2.txt', 'w')

    model = UNet(image_channels=image_channels)
    optimizer = nn.Adam(model.parameters(), lr=args.lr)

    T = args.T
    betas = np.linspace(1e-4, 0.02, T, dtype=np.float32)
    alphas = 1 - betas
    alpha_cumprod = np.cumprod(alphas)
    sqrt_alpha_cumprod = np.sqrt(alpha_cumprod)
    sqrt_one_minus_alpha_cumprod = np.sqrt(1 - alpha_cumprod)

    loss_history = []
    val_loader = lambda: load_saved_dataset(data_path, args.batch_size, split='val')

    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_loader = load_saved_dataset(data_path, args.batch_size, split='train')

        for batch_idx, (x0, _) in enumerate(train_loader):
            if x0.ndim == 3:
                x0 = x0.unsqueeze(1)
            B = x0.shape[0]
            t = np.random.randint(0, T, size=B, dtype=np.int32)
            t_var = jt.array(t)

            noise = jt.randn(x0.shape)
            sqrt_alpha_t = jt.array(sqrt_alpha_cumprod[t]).reshape((B,1,1,1))
            sqrt_one_minus_t = jt.array(sqrt_one_minus_alpha_cumprod[t]).reshape((B,1,1,1))
            x_t = sqrt_alpha_t * x0 + sqrt_one_minus_t * noise

            pred_noise = model(x_t, t_var)
            loss = ((pred_noise - noise) ** 2).mean()
            optimizer.step(loss)

            loss_val = float(loss)
            train_log.write(f"{loss_val:.6f}\n")
            loss_history.append(loss_val)

            if batch_idx % 100 == 0:
                print(f"[Epoch {epoch+1} Batch {batch_idx}] Loss: {loss_val:.6f}")

        epoch_time = time.time() - epoch_start
        perf_log.write(f"Epoch {epoch+1} Time: {epoch_time:.2f} sec\n")

        # 验证集评估
        with jt.no_grad():
            val_losses = []
            for x0, _ in val_loader():
                if x0.ndim == 3:
                    x0 = x0.unsqueeze(1)
                B = x0.shape[0]
                t = np.random.randint(0, T, size=B, dtype=np.int32)
                t_var = jt.array(t)
                noise = jt.randn(x0.shape)
                sqrt_alpha_t = jt.array(sqrt_alpha_cumprod[t]).reshape((B,1,1,1))
                sqrt_one_minus_t = jt.array(sqrt_one_minus_alpha_cumprod[t]).reshape((B,1,1,1))
                x_t = sqrt_alpha_t * x0 + sqrt_one_minus_t * noise
                pred_noise = model(x_t, t_var)
                val_loss = ((pred_noise - noise) ** 2).mean()
                val_losses.append(float(val_loss))
            avg_val_loss = np.mean(val_losses)
            print(f"[Epoch {epoch+1}] Validation Loss: {avg_val_loss:.6f}")
            perf_log.write(f"Validation Loss: {avg_val_loss:.6f}\n")

        # 保存模型和训练曲线
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
