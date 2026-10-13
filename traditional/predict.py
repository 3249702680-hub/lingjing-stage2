import sys
from pathlib import Path

import torch
from PIL import Image

from torchvision import transforms

if __package__:
    from .model import ResNet
else:
    from model import ResNet


# 与第一阶段保持相同的图片预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# 检查是否输入了图片路径
if len(sys.argv) < 2:
    print("用法: python .\\traditional\\predict.py 图片路径")
    sys.exit(1)


# 获取命令行中的图片路径
image_path = Path(sys.argv[1])

# 检查图片是否存在
if not image_path.exists():
    print("图片不存在:", image_path)
    sys.exit(1)


# 创建模型
model = ResNet(num_classes=6)

# 加载训练好的最佳模型参数
model.load_state_dict(
    torch.load(
        Path(__file__).resolve().parent / "weights" / "best_resnet.pth",
        map_location="cpu"
    )
)

# 切换到评估模式
model.eval()

# 获取类别名称
class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


# 读取图片
try:
    image = Image.open(image_path).convert("RGB")
except Exception as e:
    print("图片读取失败:", e)
    sys.exit(1)


# 使用和训练时相同的图片预处理
image = transform(image)

# 增加 batch 维度
image = image.unsqueeze(0)


# 模型推理
with torch.no_grad():
    output = model(image)

    # 将模型输出转换为各类别概率
    probabilities = torch.softmax(output, dim=1)


# 找出概率最大的类别
predicted = probabilities.argmax(dim=1).item()
predicted_class = class_names[predicted]


print("图片:", image_path)
print("预测结果:", predicted_class)

print("\n各类别概率:")

for i, class_name in enumerate(class_names):
    probability = probabilities[0][i].item()

    print(
        class_name,
        f"{probability * 100:.2f}%"
    )