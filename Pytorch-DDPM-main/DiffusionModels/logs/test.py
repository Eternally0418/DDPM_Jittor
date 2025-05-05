import matplotlib.pyplot as plt

# PyTorch 每个 epoch 总耗时（单位：秒）
pytorch_times = [
    53.70, 53.47, 53.61, 51.85, 51.89, 53.34, 51.84, 51.81, 53.69, 55.72,
    55.57, 53.95, 54.04, 52.97, 52.52, 53.39, 52.24, 54.89, 53.28, 54.06,
    54.19, 54.57, 53.74, 53.75, 53.15, 53.22, 52.13, 52.04, 52.68, 53.18,
    52.64, 53.04, 52.93, 53.34, 52.68, 53.44, 54.40, 52.60, 53.27, 54.04
]

# Jittor 每个 epoch 总耗时（单位：秒）
jittor_times = [
    144.15, 124.89, 123.93, 123.73, 124.18, 123.80, 128.44, 121.36, 118.13, 117.64,
    116.64, 125.82, 179.85, 179.23, 177.61, 176.77, 176.71, 179.27, 148.59, 73.76,
    72.87, 73.61, 72.50, 72.13, 73.04, 73.50, 74.30, 73.14, 76.39, 74.04,
    72.90, 74.18, 73.44, 72.84, 73.12, 73.12, 73.04, 73.59, 73.96, 73.15
]

# 生成 epoch 编号
epochs = list(range(1, 41))

# 绘图
plt.figure(figsize=(14, 6))
plt.plot(epochs, jittor_times, label='Jittor', color='orange', marker='o')
plt.plot(epochs, pytorch_times, label='PyTorch', color='red', marker='s')
plt.xlabel("Epoch")
plt.ylabel("Total Time (seconds)")
plt.title("Epoch-wise Total Training Time: Jittor vs PyTorch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("epoch_time_compare_english.png", dpi=300)
plt.show()
