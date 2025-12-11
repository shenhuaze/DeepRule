#!/usr/bin/env python3
"""
创建markdown文件，展示原图和提取数据的对比图片
用法: python create_image_markdown.py [--original_dir <dir>] [--extracted_dir <dir>]
"""

import os
import glob
import argparse
from collections import defaultdict

def create_image_markdown(original_dir, extracted_dir):
    """
    创建markdown文件，对比原图和提取数据后的图片

    Args:
        original_dir (str): 原始图片目录路径
        extracted_dir (str): 提取数据后的图片目录路径
    """
    # 检查目录是否存在
    if not os.path.exists(original_dir):
        print(f"错误：目录 {original_dir} 不存在")
        return

    if not os.path.exists(extracted_dir):
        print(f"错误：目录 {extracted_dir} 不存在")
        return

    # 将markdown文件放在与original_dir同级目录，文件名固定为"image_comparison.md"
    original_parent_dir = os.path.dirname(os.path.abspath(original_dir))
    markdown_path = os.path.join(original_parent_dir, "image_comparison.md")

    # 获取原图列表
    original_images = glob.glob(os.path.join(original_dir, "*.png"))

    # 按前缀分组
    # 前缀定义为：从文件名开始到最后一个"-"之前的部分
    prefix_groups = defaultdict(list)

    for img_path in original_images:
        filename = os.path.basename(img_path)
        # 找到最后一个"-"的位置
        last_dash_index = filename.rfind('-')
        if last_dash_index > 0:
            prefix = filename[:last_dash_index]
        else:
            prefix = filename
        prefix_groups[prefix].append(filename)

    # 按前缀字母顺序排序
    sorted_prefixes = sorted(prefix_groups.keys())

    # 计算相对路径（从markdown文件位置到图片目录）
    markdown_dir = os.path.dirname(os.path.abspath(markdown_path))
    original_relative_path = os.path.relpath(original_dir, markdown_dir)
    extracted_relative_path = os.path.relpath(extracted_dir, markdown_dir)

    # 创建markdown文件
    with open(markdown_path, 'w', encoding='utf-8') as f:
        # 写入标题
        f.write("# 图表数据提取对比\n\n")
        f.write("本文档展示了原始图表图片与数据提取后生成的曲线图的对比。\n\n")
        f.write(f"- 原始图片目录: `{original_relative_path}`\n")
        f.write(f"- 提取数据图片目录: `{extracted_relative_path}`\n\n")
        f.write("---\n\n")

        # 遍历每个前缀组
        for prefix in sorted_prefixes:
            original_filenames = sorted(prefix_groups[prefix])

            # 对于每个前缀的每个文件
            for filename in original_filenames:
                # 构建对应的提取数据文件名
                extracted_filename = filename.replace('.png', '_extracted.png')
                # 使用相对路径
                original_display_path = os.path.join(original_relative_path, filename)
                extracted_display_path = os.path.join(extracted_relative_path, extracted_filename)
                # 用于检查文件是否存在的绝对路径
                original_absolute_path = os.path.join(original_dir, filename)
                extracted_absolute_path = os.path.join(extracted_dir, extracted_filename)

                # 写入文件名
                f.write(f"### 图片名：{filename}\n\n")

                # 写入原图
                f.write("原图：\n\n")
                f.write(f'<img src="{original_display_path}" alt="{filename}" width="500">\n\n')

                # 写入提取数据的图（如果存在）
                if os.path.exists(extracted_absolute_path):
                    f.write("数据提取结果：\n\n")
                    f.write(f'<img src="{extracted_display_path}" alt="{extracted_filename}" width="500">\n\n')
                else:
                    f.write("数据提取结果：*（未找到对应的提取数据图片）*\n\n")

                # 添加分隔线（除了最后一个）
                f.write("---\n\n")

    print(f"Markdown文件已生成: {markdown_path}")
    print(f"共处理了 {len(sorted_prefixes)} 个前缀组")
    total_files = sum(len(files) for files in prefix_groups.values())
    print(f"总文件数: {total_files}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='创建markdown文件，对比原图和提取数据的图片')
    parser.add_argument('--original_dir', default='data/sample_images',
                       help='原始图片目录路径')
    parser.add_argument('--extracted_dir', default='data/sample_images_extracted_plot',
                       help='提取数据后的图片目录路径')

    args = parser.parse_args()

    # 创建markdown文件
    create_image_markdown(args.original_dir, args.extracted_dir)


if __name__ == "__main__":
    main()