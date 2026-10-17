# 灵境 AI 第二阶段：垃圾视觉识别对比

## 项目简介

本项目针对同一组垃圾图片，分别使用两种方案进行处理：

- **传统视觉方案：YOLO 目标检测**
- **多模态方案：Qwen3-VL**

最终比较两种方案在目标检测、垃圾分类、语义理解和适用场景上的差异。

## 项目结构

```text
lingjing-stage2/
├── compare.py
├── images/
│   ├── paper.jpg
│   └── plastic.jpg
├── traditional/
│   ├── yolo_detect.py
│   ├── opencv_demo.py
│   └── weights/
│       └── garbage_detect.pt
├── multimodal/
│   └── api_predict.py
├── results/
│   ├── yolo/
│   ├── multimodal/
│   └── comparison/
├── README.md
└── requirements.txt
```

## YOLO 方案

YOLO 对输入图片进行目标检测，输出：

```text
类别
置信度
Bounding Box
```

例如 `paper.jpg`：

```text
类别：paper
置信度：0.9664
边界框：(34.89, 17.97, 189.86, 113.73)
```

YOLO 适合需要快速检测和精确目标定位的任务。

## Qwen3-VL 方案

Qwen3-VL 接收：

```text
图片 + Prompt
```

模型回答：

1. 图片中的物体是什么
2. 属于什么垃圾类别
3. 是否可回收
4. 判断理由

Qwen3-VL 更适合图片描述、垃圾分类解释和视觉问答。

## 两种方案对比

| 对比项目 | YOLO | Qwen3-VL |
|---|---|---|
| 类别识别 | 支持 | 支持 |
| 置信度 | 支持 | 不作为主要输出 |
| Bounding Box | 支持 | 不擅长 |
| 图片描述 | 不擅长 | 支持 |
| 判断理由 | 不擅长 | 支持 |
| 适用场景 | 实时检测、目标定位 | 复杂场景、描述、问答 |

实验中：

```text
paper.jpg
YOLO → paper，置信度 0.9664
Qwen3-VL → 纸张，可回收物，并给出解释

plastic.jpg
YOLO → 未检测到目标
Qwen3-VL → 塑料饮料瓶，可回收物，并给出解释
```

## OpenCV

项目使用 OpenCV 完成了：

```text
图片读取
图片缩放
图片裁剪
图片显示
图片保存
```

示例代码：

```text
traditional/opencv_demo.py
```

## 安装依赖

```powershell
pip install -r requirements.txt
```

## 配置 API Key

```powershell
setx DASHSCOPE_API_KEY "你的API_Key"
```

设置完成后重新打开终端。

API Key 不应写入代码或上传到 GitHub。

## 运行 YOLO

```powershell
python traditional\yolo_detect.py "images/paper.jpg"
```

## 运行 Qwen3-VL

```powershell
python multimodal\api_predict.py "images/paper.jpg"
```

## 运行完整对比

```powershell
python compare.py "images/paper.jpg"
```

或者：

```powershell
python compare.py "images/plastic.jpg"
```

结果保存到：

```text
results/comparison/
```

## 总结

YOLO 更适合固定类别、快速检测和目标定位。

Qwen3-VL 更擅长图片语义理解、自然语言描述和判断理由解释。

两种方案各有优势，可以根据具体视觉任务选择使用。