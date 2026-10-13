# 灵境 AI 第二阶段：垃圾识别方案对比

## 项目目标

对同一个垃圾识别任务分别使用传统深度学习和多模态大模型方案，在相同测试图片和类别定义下比较分类结果、准确率与耗时。

## 两种方案

- **传统深度学习方案**：已迁移第一阶段手写 ResNet 的结构和已训练权重，可对新图片进行分类预测。模型结构、图片预处理和类别顺序保持一致，不重新训练，不修改第一阶段项目。
- **多模态大模型方案**：后续通过视觉大模型 API 输入图片并获取垃圾类别。`multimodal/api_predict.py` 当前仍为空，API 调用尚未实现。

## 目录结构

```text
lingjing-stage2/
├── traditional/
│   ├── model.py
│   ├── predict.py
│   └── weights/
│       └── best_resnet.pth
├── multimodal/
│   └── api_predict.py
├── images/
├── results/
├── README.md
├── requirements.txt
└── .gitignore
```

## 传统方案运行方式

在项目根目录打开 PowerShell，准备独立虚拟环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

准备一张测试图片，将下方示例路径替换为实际图片路径，再运行：

```powershell
.\.venv\Scripts\python.exe traditional\predict.py .\images\example.jpg
```

也可以使用模块方式：

```powershell
.\.venv\Scripts\python.exe -m traditional.predict .\images\example.jpg
```

权重从脚本所在目录的 `weights/best_resnet.pth` 加载，即项目内的 `traditional/weights/best_resnet.pth`；不依赖当前工作目录。图片路径可以使用绝对路径，相对路径以当前工作目录为基准。

图片预处理保持第一阶段的 `Resize((224, 224))` 和 `ToTensor()`。六类顺序为：`cardboard`、`glass`、`metal`、`paper`、`plastic`、`trash`。

## 多模态方案后续运行方式

待实现 API 预测逻辑、选择视觉 API 服务，并通过环境变量配置 API 密钥后，再运行 `multimodal/api_predict.py`。

本次仅迁移已训练的传统方案，未重新训练、调用 API 或下载数据。
