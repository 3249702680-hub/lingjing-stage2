import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_script(script_path, image_path):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(image_path)
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env
    )

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("子程序运行失败")

    return result.stdout


def main():
    parser = argparse.ArgumentParser(
        description="YOLO 与 Qwen3-VL 垃圾识别对比"
    )

    parser.add_argument(
        "image",
        help="待分析图片，例如 images/paper.jpg"
    )

    args = parser.parse_args()

    image_path = Path(args.image).resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在：{image_path}")

    yolo_script = Path("traditional/yolo_detect.py").resolve()
    qwen_script = Path("multimodal/api_predict.py").resolve()

    print("=" * 60)
    print("垃圾视觉识别对比实验")
    print("=" * 60)

    print("\n输入图片：")
    print(image_path)

    print("\n" + "=" * 60)
    print("方案一：YOLO")
    print("=" * 60)

    yolo_result = run_script(
        yolo_script,
        image_path
    )

    print(yolo_result)

    print("\n" + "=" * 60)
    print("方案二：Qwen3-VL")
    print("=" * 60)

    qwen_result = run_script(
        qwen_script,
        image_path
    )

    print(qwen_result)

    output_dir = Path("results/comparison")
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = output_dir / f"{image_path.stem}_comparison.txt"

    text = f"""
输入图片：
{image_path}

================ YOLO ================

{yolo_result}

================ Qwen3-VL ================

{qwen_result}
"""

    output_file.write_text(
        text,
        encoding="utf-8"
    )

    print("=" * 60)
    print("对比完成")
    print("结果保存到：", output_file.resolve())


if __name__ == "__main__":
    main()