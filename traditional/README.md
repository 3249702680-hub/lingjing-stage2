# 垃圾物品 YOLO 目标检测方案

当前入口为 `yolo_detect.py`，默认加载项目内 `traditional/weights/garbage_detect.pt`，只对单张图片执行目标检测，不进行训练。

## 方案职责

- **YOLO 负责目标检测**：定位图片中的垃圾物品，输出类别、置信度与边界框，并保存标注图。
- **大模型负责语义理解**：结合图片中的物品与场景进行语义理解和解释。此职责说明不改变 `multimodal/` 的代码。

## 安装依赖

在第二阶段项目根目录打开 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

依赖使用现有 `requirements.txt` 中的 `ultralytics`、`Pillow` 等库，本次不新增依赖。

## 默认权重与运行方式

将已训练好的、兼容 Ultralytics 的垃圾目标检测 `.pt` 权重保存为：

```text
traditional/weights/garbage_detect.pt
```

这个文件需要是真正的 YOLO 目标检测权重。仅重命名 COCO 或 ResNet 权重不会获得垃圾检测能力。默认权重缺失时，程序会提示完整路径和处理方法，不自动回退到 COCO 模型。

用实际图片路径替换下方示例：

```powershell
.\.venv\Scripts\python.exe traditional\yolo_detect.py .\images\example.jpg
```

模块方式同样支持：

```powershell
.\.venv\Scripts\python.exe -m traditional.yolo_detect .\images\example.jpg
```

## 使用其他权重覆盖默认模型

`--model` 参数继续有效，可以指定另一份垃圾检测权重：

```powershell
.\.venv\Scripts\python.exe traditional\yolo_detect.py .\images\example.jpg --model .\traditional\weights\other_garbage.pt
```

也可以传入本地 `.pt` 文件的绝对路径；路径含空格时加引号。即使默认权重缺失，也可通过 `--model` 选择其他可用权重。模型必须是 `detect` 任务，脚本不接受分类、分割或姿态模型。

可选参数保持不变：

- `--conf`：置信度阈值，默认 `0.25`，范围为 `0` 到 `1`。
- `--device`：默认 `cpu`；有兼容 GPU 环境时可以填 `0`。
- `--output-dir`：默认项目内 `results/yolo`。

默认权重与默认输出目录以脚本位置为基准。输入图片和手动指定的相对路径以当前工作目录为基准。

## 检测流程与输出

1. 校验参数和默认权重，并读取一张本地 RGB 图片。
2. 加载选定的 YOLO 检测模型，调用 `model.predict()`。
3. 保存带类别、置信度和边界框的 JPG 图片。
4. 输出每个目标的类别、置信度及原图像素坐标 `(x1, y1, x2, y2)`。

类别从所加载权重的 `result.names` 读取，不硬编码。左上角为 `(x1, y1)`，右下角为 `(x2, y2)`。结果文件名带随机后缀，不覆盖输入图或已有结果；没有目标时也会保存图片并给出提示。

原 ResNet 文件、`weights/best_resnet.pth`、`weights/yolo11n.pt` 和 Git 历史继续保留。本次不修改检测逻辑、不修改 `multimodal/`，不训练、不提交 Git commit。
