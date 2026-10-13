import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 第一层卷积：可改变通道数和高宽
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # 第二层卷积：保持当前通道数和尺寸
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果主分支和 shortcut 的 shape 不一致，就用 1×1 卷积调整
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # shape 一致时，shortcut 直接保留原输入
            self.shortcut = nn.Identity()

    def forward(self, x):
        # shortcut 分支
        identity = self.shortcut(x)

        # 主分支
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # ResNet 核心：F(x) + x
        out = out + identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        # 当前输入通道数，后面创建 layer 时会更新
        self.in_channels = 64

        # ResNet 开头的大卷积
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # 最大池化，继续缩小特征图高宽
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        # 4 个残差阶段，每个阶段有 2 个 ResidualBlock
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)

        # 把每个通道的特征图压成 1×1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 512 个特征最终输出 6 类分数
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        blocks = []

        # 第一个 block 负责改变通道数和可能的高宽
        blocks.append(
            ResidualBlock(
                self.in_channels,
                out_channels,
                stride
            )
        )

        # 前一个 block 的输出通道数，就是后一个 block 的输入通道数
        self.in_channels = out_channels

        # 后面的 block 保持尺寸不变，所以 stride=1
        for _ in range(1, num_blocks):
            blocks.append(
                ResidualBlock(
                    self.in_channels,
                    out_channels,
                    stride=1
                )
            )

        # 把多个 ResidualBlock 按顺序串起来
        return nn.Sequential(*blocks)

    def forward(self, x):
        # ResNet 开头
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 4 个残差阶段
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 池化并压平
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # 输出 6 类分数
        x = self.fc(x)

        return x

if __name__ == "__main__":
    # 模拟 2 张 224×224 RGB 图片测试模型
    model = ResNet(num_classes=6)
    x = torch.randn(2, 3, 224, 224)

    output = model(x)

    print("输入 shape:", x.shape)
    print("输出 shape:", output.shape)
    print("模型输出:")
    print(output)