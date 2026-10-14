"""使用 Ultralytics YOLO 对一张本地图片进行目标检测，只执行推理。"""

import argparse
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "traditional" / "weights" / "garbage_detect.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="检测一张图片中的目标，输出类别、置信度和边界框，并保存标注图。"
    )
    parser.add_argument("image", type=Path, help="本地图片路径")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="垃圾目标检测 .pt 权重路径；默认使用 traditional/weights/garbage_detect.pt",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="置信度阈值，范围 0 到 1"
    )
    parser.add_argument("--device", default="cpu", help="推理设备，默认 cpu；GPU 可填 0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "yolo",
        help="结果图片目录，默认是项目下的 results/yolo",
    )
    args = parser.parse_args()
    args.image = args.image.expanduser().resolve()
    args.model = args.model.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.image.is_file():
        parser.error(f"图片文件不存在：{args.image}")
    if not 0 <= args.conf <= 1:
        parser.error("--conf 必须在 0 到 1 之间")
    if args.model.suffix.lower() != ".pt":
        parser.error("--model 请提供预训练的 .pt 权重，不能使用未训练的模型配置")
    if args.model == DEFAULT_MODEL.resolve() and not args.model.is_file():
        parser.error(
            f"默认垃圾检测权重不存在：{args.model}。"
            "请将垃圾 YOLO 检测权重放到 traditional/weights/garbage_detect.pt，"
            "或用 --model 指定现有的目标检测 .pt 文件。"
        )
    return args


def main():
    args = parse_args()
    try:
        from PIL import Image

        # 先解码本地图片，保证只处理一张图片；不接受视频、目录或 URL。
        with Image.open(args.image) as source_image:
            image = source_image.convert("RGB")
    except ImportError:
        print("缺少 Pillow，请先安装项目 requirements.txt 中的依赖。", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"无法读取图片：{error}", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError as error:
        print(f"无法导入 Ultralytics，请先安装依赖：{error}", file=sys.stderr)
        return 1

    try:
        # 加载所选 YOLO 权重；本脚本不调用 train。
        model = YOLO(str(args.model))
        if model.task != "detect":
            raise ValueError("请使用目标检测权重，不能使用分类、分割或姿态模型")

        result = model.predict(
            source=image,
            conf=args.conf,
            device=args.device,
            save=False,
            verbose=False,
        )[0]

        args.output_dir.mkdir(parents=True, exist_ok=True)
        # 原子创建新的文件名，重复运行也不会覆盖已有图片。
        with NamedTemporaryFile(
            prefix=f"{args.image.stem}_detected_",
            suffix=".jpg",
            dir=args.output_dir,
            delete=False,
        ) as output_file:
            output_path = Path(output_file.name)
        result.save(filename=str(output_path))
        if output_path.stat().st_size == 0:
            raise OSError("标注图片保存失败")

        print("图片：", args.image)
        print("模型：", args.model)
        boxes = result.boxes
        print("检测数量：", len(boxes) if boxes is not None else 0)
        if boxes is None or len(boxes) == 0:
            print("未检测到达到置信度阈值的目标，仍已保存结果图片。")
        else:
            for index, box in enumerate(boxes, start=1):
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(
                    f"[{index}] 类别={result.names[class_id]} "
                    f"置信度={confidence:.4f} "
                    f"边界框(x1, y1, x2, y2)=("
                    f"{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})"
                )
        print("结果图片：", output_path)
        return 0
    except Exception as error:
        print(f"目标检测失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
