import os
import argparse
from pathlib import Path
from http import HTTPStatus

from dashscope import MultiModalConversation


def main():
    # 1. 接收命令行中的图片路径
    parser = argparse.ArgumentParser(
        description="使用 Qwen3-VL 分析垃圾图片"
    )

    parser.add_argument(
        "image",
        help="待分析图片路径，例如 images/paper.jpg"
    )

    args = parser.parse_args()


    # 2. 获取 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "没有找到 DASHSCOPE_API_KEY 环境变量"
        )


    # 3. 获取图片路径
    image_path = Path(args.image).resolve()

    if not image_path.exists():
        raise FileNotFoundError(
            f"图片不存在：{image_path}"
        )

    image_url = f"file://{image_path.as_posix()}"


    # 4. Prompt
    prompt = """
请分析图片中的主要垃圾物体。

请回答：
1. 物体是什么？
2. 它属于哪种垃圾类别？
3. 是否可回收？
4. 简单说明判断理由。
"""


    # 5. 多模态输入
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_url},
                {"text": prompt}
            ]
        }
    ]


    # 6. 调用大模型
    response = MultiModalConversation.call(
        api_key=api_key,
        model="qwen3-vl-plus",
        messages=messages
    )


    # 7. 处理返回结果
    if response.status_code == HTTPStatus.OK:

        answer = response.output.choices[0].message.content[0]["text"]

        print("图片：", image_path)
        print("模型：qwen3-vl-plus")

        print("\n模型回答：")
        print(answer)


        # 8. 创建结果目录
        output_dir = Path("results/multimodal")

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        # 9. 生成结果文件名
        output_file = output_dir / f"{image_path.stem}_result.txt"


        # 10. 保存结果
        output_file.write_text(
            answer,
            encoding="utf-8"
        )

        print("\n结果已保存：", output_file.resolve())

    else:

        print("调用失败")
        print("状态码：", response.status_code)
        print("错误代码：", response.code)
        print("错误信息：", response.message)


if __name__ == "__main__":
    main()