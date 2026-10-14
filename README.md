# 灵境 AI 第二阶段：传统视觉与多模态方案对比

## 项目目标

使用传统视觉方案和多模态大模型方案处理同一批垃圾相关图片。当前传统方案的主入口已改为 YOLO 预训练目标检测：定位图片中的物体，输出类别、置信度和边界框，并保存标注图片。

## 两种方案

- **传统视觉方案**：使用 Ultralytics YOLO11n 预训练模型进行单张图片检测，不训练模型。运行说明见 [traditional/README.md](traditional/README.md)。原 ResNet 文件与权重保留作为历史分类方案。
- **多模态大模型方案**：保留 `multimodal/` 的原有内容，本次未修改该目录或调用 API。

默认 YOLO11n 输出 COCO 通用物体类别，与原 ResNet 六类垃圾分类不同。开展两种方案对比前，应统一检测类别与评价方式。

## 目录结构

```text
lingjing-stage2/
├── traditional/
│   ├── yolo_detect.py         # 当前 YOLO 检测入口
│   ├── README.md              # YOLO 运行说明
│   ├── model.py               # 保留的 ResNet 模型
│   ├── predict.py             # 保留的 ResNet 分类入口
│   └── weights/
│       ├── best_resnet.pth    # 保留的 ResNet 权重
│       └── yolo11n.pt         # 首次推理缺失时自动下载
├── multimodal/
│   └── api_predict.py
├── images/
├── results/
├── README.md
├── requirements.txt
└── .gitignore
```

## 安装与运行

在项目根目录打开 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe traditional\yolo_detect.py .\images\example.jpg
```

请用实际图片路径替换 `example.jpg`。默认结果保存到项目内 `results/yolo/`；终端输出每个目标的类别、置信度和原图像素边界框。首次运行缺少默认权重时需要联网下载预训练模型；不会下载训练数据或执行训练。

依赖新增 `ultralytics` 与 `Pillow`，其余原有依赖保留；详情见传统方案 README。本次不删除原文件，不改写 Git 历史，不提交 Git commit。
