#!/usr/bin/env python3
"""
绘图脚本：从JSON文件中读取多条曲线数据并绘制在同一张图中
JSON格式：annotations包含多个对象，每个对象的bbox字段包含一系列点的坐标
"""

import json
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np


def plot_curves_from_json(json_file):
    """
    从JSON文件读取曲线数据并绘制

    Args:
        json_file: JSON数据文件路径
    """
    # 读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 获取JSON文件的目录
    json_dir = os.path.dirname(json_file)
    if not json_dir:
        json_dir = '.'

    # 获取文件名并生成输出文件名
    original_filename = data.get('file_name', '')
    if original_filename:
        # 在.png前添加"_plot"
        if '.png' in original_filename:
            output_filename = original_filename.replace('.png', '_plot.png')
        else:
            # 如果没有.png后缀，直接添加_plot.png
            base_name = os.path.splitext(original_filename)[0]
            output_filename = f"{base_name}_plot.png"
    else:
        # 如果没有file_name字段，使用JSON文件名
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        output_filename = f"{base_name}_plot.png"

    # 构建完整的输出路径
    output_path = os.path.join(json_dir, output_filename)

    # 创建图像
    plt.figure(figsize=(12, 8))

    # 获取图像尺寸（如果有的话）
    img_height = data.get('height', None)
    img_width = data.get('width', None)

    # 遍历annotations中的每条曲线
    annotations = data.get('annotations', [])

    # 收集所有数据点的范围
    all_x = []
    all_y = []

    for i, annotation in enumerate(annotations):
        bbox = annotation.get('bbox', [])

        if not bbox:
            continue

        # 将bbox中的坐标转换为x和y的列表
        # bbox格式：[x1, y1, x2, y2, x3, y3, ...]
        x_coords = []
        y_coords = []

        for j in range(0, len(bbox), 2):
            if j + 1 < len(bbox):
                x_coords.append(bbox[j])
                y_coords.append(bbox[j + 1])
                # 收集所有点用于计算范围
                all_x.append(bbox[j])
                all_y.append(bbox[j + 1])

        # 绘制曲线
        plt.plot(x_coords, y_coords, marker='o', markersize=3,
                label=f'Curve {i+1} (ID: {annotation.get("id", i)})',
                linewidth=2)

    # 设置图像属性
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Curves from {original_filename}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 设置坐标轴范围，基于实际数据范围，稍微扩展一点
    if all_x and all_y:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

        # 计算范围并稍微扩展5%
        x_range = x_max - x_min
        y_range = y_max - y_min

        # 如果所有点的x或y坐标相同，设置一个默认范围
        if x_range == 0:
            x_range = 10
        if y_range == 0:
            y_range = 10

        x_padding = x_range * 0.05
        y_padding = y_range * 0.05

        plt.xlim(x_min - x_padding, x_max + x_padding)
        # Y坐标从上到下逐渐增大（图像坐标系）
        plt.ylim(y_max + y_padding, y_min - y_padding)
    elif img_width and img_height:
        # 如果没有数据点，但有图像尺寸，使用图像尺寸
        plt.xlim(0, img_width)
        # Y坐标从上到下逐渐增大（图像坐标系）
        plt.ylim(img_height, 0)

    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存为: {output_path}")

    # 关闭图形，不显示
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='从JSON文件绘制曲线')
    parser.add_argument('json_file', help='包含曲线数据的JSON文件路径')

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.json_file):
        print(f"错误：文件 '{args.json_file}' 不存在")
        return

    try:
        plot_curves_from_json(args.json_file)
    except Exception as e:
        print(f"处理文件时出错: {e}")


if __name__ == '__main__':
    main()