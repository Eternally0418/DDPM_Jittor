import re

# 替换为你的日志文件路径
file_path = "performance_log2_pytorch.txt"

# 读取文件内容
with open(file_path, "r", encoding="utf-8") as file:
    content = file.read()

# 使用正则表达式提取所有 Validation Loss 数值
validation_loss_values = re.findall(r'Validation Loss:\s*([0-9]*\.?[0-9]+)', content)

# 提取前 40 个并转换为浮点数
validation_loss_values = [float(value) for value in validation_loss_values[:40]]

# 输出所有提取的值
for i, val in enumerate(validation_loss_values, start=1):
    print(f"{i}. {val}")
