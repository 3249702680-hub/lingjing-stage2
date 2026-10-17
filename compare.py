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
        print("子程序运行失败：")
        print(result.stderr)
        raise RuntimeError("子程序运行失败")

    return result.stdout


def main():
    # 1. 命令行参数
    parser = argparse.ArgumentParser(
        description="YOLO 与 Qwen3-VL 垃圾视觉识别对比"
    )

    parser.add_argument(
        "image",
        help="待分析图片，例如 images/paper.jpg"
    )

    args = parser.parse_args()

    # 2. 图片路径
    image_path = Path(args.image).resolve()

    if not image_path.exists():
        raise FileNotFoundError(
            f"图片不存在：{image_path}"
        )

    # 3. 两个方案的程序
    yolo_script = Path(
        "traditional/yolo_detect.py"
    ).resolve()

    qwen_script = Path(
        "multimodal/api_predict.py"
    ).resolve()

    # 4. 项目标题
    print("=" * 60)
    print("垃圾视觉识别对比实验")
    print("=" * 60)

    print("\n输入图片：")
    print(image_path)

    # 5. YOLO
    print("\n" + "=" * 60)
    print("方案一：YOLO 目标检测")
    print("=" * 60)

    yolo_result = run_script(
        yolo_script,
        image_path
    )

    print(yolo_result)

    # 6. Qwen3-VL
    print("\n" + "=" * 60)
    print("方案二：Qwen3-VL 多模态理解")
    print("=" * 60)

    qwen_result = run_script(
        qwen_script,
        image_path
    )

    print(qwen_result)

    # 7. 两种方案对比
    print("\n" + "=" * 60)
    print("两种方案对比")
    print("=" * 60)

    comparison_text = (
        "YOLO：\n"
        "- 输出类别、置信度和 Bounding Box\n"
        "- 可以精确定位目标\n"
        "- 推理速度快\n"
        "- 适合固定类别、实时目标检测任务\n"
        "- 效果受训练数据和模型类别范围影响\n"
        "\n"
        "Qwen3-VL：\n"
        "- 可以识别和理解图片内容\n"
        "- 可以回答垃圾分类问题\n"
        "- 可以使用自然语言解释判断原因\n"
        "- 更适合复杂场景、描述和视觉问答\n"
        "- 不擅长像 YOLO 一样输出精确 Bounding Box\n"
    )

    print(comparison_text)

    # 8. 创建结果目录
    output_dir = Path("results/comparison")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # 9. 保存 TXT
    txt_file = (
        output_dir
        / f"{image_path.stem}_comparison.txt"
    )

    txt_content = (
        "垃圾视觉识别对比实验\n\n"
        f"输入图片：\n{image_path}\n\n"
        "================ YOLO ================\n\n"
        f"{yolo_result}\n"
        "================ Qwen3-VL ================\n\n"
        f"{qwen_result}\n"
        "================ 两种方案对比 ================\n\n"
        f"{comparison_text}"
    )

    txt_file.write_text(
        txt_content,
        encoding="utf-8"
    )

    # 10. Markdown 并排对比
    md_file = (
        output_dir
        / f"{image_path.stem}_comparison.md"
    )

    md_lines = [
        "# 垃圾视觉识别对比实验",
        "",
        "## 输入图片",
        "",
        f"`{image_path.name}`",
        "",
        "## 两种方案并排对比",
        "",
        "| 对比项目 | YOLO | Qwen3-VL |",
        "|---|---|---|",
        "| 核心任务 | 目标检测 | 多模态视觉理解 |",
        "| 类别识别 | 支持 | 支持 |",
        "| 置信度 | 支持 | 不作为主要输出 |",
        "| Bounding Box | 支持 | 不擅长精确定位 |",
        "| 自然语言描述 | 不擅长 | 支持 |",
        "| 判断理由 | 不擅长 | 支持 |",
        "| 推理特点 | 快、输出结构明确 | 语义理解能力强 |",
        "| 适用场景 | 固定类别、实时检测、目标定位 | 复杂场景、垃圾分类解释、图片问答 |",
        "",
        "## YOLO 运行结果",
        "",
        "```text",
        yolo_result.strip(),
        "```",
        "",
        "## Qwen3-VL 运行结果",
        "",
        "```text",
        qwen_result.strip(),
        "```",
        "",
        "## 使用体验与适用场景",
        "",
        "YOLO 可以直接输出类别、置信度和目标边界框，",
        "结果结构清晰，适合固定类别、需要快速检测和精确定位的视觉任务。",
        "",
        "但是 YOLO 的效果与训练数据关系较大，",
        "如果输入图片与训练数据差异较大，可能出现漏检或者误判。",
        "",
        "Qwen3-VL 更擅长理解图片中的语义信息，",
        "不仅能够识别物体，还能够判断垃圾类别，并通过自然语言解释判断理由。",
        "",
        "因此，Qwen3-VL 更适合复杂场景、视觉问答、图片描述和需要解释的任务。",
        "但它不像 YOLO 一样能够稳定输出精确的目标 Bounding Box。",
        ""
    ]

    md_content = "\n".join(md_lines)

    md_file.write_text(
        md_content,
        encoding="utf-8"
    )

    # 11. 完成
    print("\n" + "=" * 60)
    print("对比实验完成")
    print("=" * 60)

    print("TXT结果：")
    print(txt_file.resolve())

    print("\nMarkdown对比：")
    print(md_file.resolve())


if __name__ == "__main__":
    main()