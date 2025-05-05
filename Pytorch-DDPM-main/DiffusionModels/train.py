import os, time, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from model import UNet

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_npz_dataset(path, batch_size):
    data = np.load(path)
    images, labels = data['images'], data['labels']
    indices = np.random.permutation(len(images))
    images, labels = images[indices], labels[indices]
    images = torch.tensor(images, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(images, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = args.dataset
    data_path = f"data/{dataset_name}_processed.npz"
    image_channels = args.image_channels if args.image_channels is not None else (1 if dataset_name == "mnist" else 3)

    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    perf_log = open('logs/performance_log2_pytorch.txt', 'w')
    train_log = open('logs/train_log_pytorch2.txt', 'w')

    model = UNet(image_channels=image_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    T = args.T
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)

    loss_history = []

    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_loader = load_npz_dataset(data_path, args.batch_size)
        model.train()

        for batch_idx, (x0, _) in enumerate(train_loader):
            batch_start = time.time()
            x0 = x0.to(device)
            if x0.ndim == 3:
                x0 = x0.unsqueeze(1)
            B = x0.size(0)

            t = torch.randint(0, T, (B,), device=device)
            sqrt_alpha_t = sqrt_alpha_cumprod[t].view(B, 1, 1, 1)
            sqrt_one_minus_t = sqrt_one_minus_alpha_cumprod[t].view(B, 1, 1, 1)
            noise = torch.randn_like(x0)
            x_t = sqrt_alpha_t * x0 + sqrt_one_minus_t * noise

            pred_noise = model(x_t, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            loss_history.append(loss_val)
            train_log.write(f"{loss_val:.6f}\n")

            batch_time = time.time() - batch_start
            perf_log.write(f"Epoch {epoch+1} Batch {batch_idx+1} Time: {batch_time:.3f} sec\n")

            if batch_idx % 100 == 0:
                print(f"[Epoch {epoch+1} Batch {batch_idx}] Loss: {loss_val:.6f}")

        # 每轮后保存模型与 loss 曲线
        torch.save(model.state_dict(), f"checkpoints/ddpm_{dataset_name}_epoch{epoch+1}.pth")

        plt.figure()
        plt.plot(loss_history)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.savefig(f"logs/loss_curve_{dataset_name}.png")
        plt.close()

        epoch_time = time.time() - epoch_start
        perf_log.write(f"Epoch {epoch+1} Total Time: {epoch_time:.2f} sec\n")
        perf_log.flush()
        train_log.flush()

    torch.save(model.state_dict(), f"ddpm_{dataset_name}.pth")
    print("训练完成")
    perf_log.close()
    train_log.close()

if __name__ == '__main__':
    main()
