# 灵境 AI 第二阶段：垃圾识别方案对比

## 项目目标

对同一个垃圾识别任务分别使用传统深度学习和多模态大模型方案，在相同测试图片和类别定义下比较分类结果、准确率与耗时。

## 两种方案

- **传统深度学习方案**：后续复用第一阶段手写 ResNet 的结构和已训练权重，对新图片进行分类预测。本项目不实现模型训练，也不修改第一阶段项目。
- **多模态大模型方案**：后续通过视觉大模型 API 输入图片并获取垃圾类别。具体 API 服务与调用逻辑待后续实现。

## 目录结构

```text
lingjing-stage2/
├── traditional/
│   ├── model.py          # 空文件，后续复用第一阶段 ResNet
│   └── predict.py        # 空文件，后续实现传统方案预测
├── multimodal/
│   └── api_predict.py    # 空文件，后续实现视觉大模型 API 预测
├── images/               # 空目录，后续放入两种方案共用的测试图片
├── results/              # 空目录，后续保存预测结果与对比记录
├── README.md
├── requirements.txt
└── .gitignore
```

## 后续运行方式

当前仅为项目骨架，三个 Python 文件均为空，尚无可用的预测功能。

后续先准备测试图片、第一阶段模型权重，以及所选视觉 API 的访问配置；API 密钥通过环境变量提供。完成预测逻辑后，在项目根目录打开 PowerShell，执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe traditional\predict.py
.\.venv\Scripts\python.exe multimodal\api_predict.py
```

上述命令为后续实现完成后的运行方式。本次仅创建骨架，不安装依赖、不训练模型、不调用 API、不下载数据。
