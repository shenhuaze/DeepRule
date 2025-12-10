#!/usr/bin/env python3
"""
绘图脚本：从JSON文件中读取line_data字段的多条曲线数据并绘制在同一张图中
JSON格式：包含line_data字段，其中每个列表代表一条曲线的数据点
用法: python plot_extracted_data.py <json_file>
"""

import json
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np


def plot_curves_from_json(json_file):
    """
    从JSON文件读取line_data并绘制多条曲线

    Args:
        json_file (str): JSON文件路径
    """
    # 读取JSON文件
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {json_file}")
        return
    except json.JSONDecodeError:
        print(f"错误: 无法解析JSON文件 {json_file}")
        return

    # 检查是否包含line_data
    if 'line_data' not in data:
        print(f"错误: JSON文件中没有找到'line_data'字段")
        return

    line_data = data['line_data']

    # 获取JSON文件的目录
    json_dir = os.path.dirname(json_file)
    if not json_dir:
        json_dir = '.'

    # 生成输出文件名（将.json替换为_plot.png）
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    output_filename = f"{base_name}_plot.png"
    output_path = os.path.join(json_dir, output_filename)

    # 设置图像
    plt.figure(figsize=(12, 8))

    # 为每条曲线生成不同的颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(line_data)))

    # 绘制每条曲线
    for i, curve_data in enumerate(line_data):
        # 将数据转换为numpy数组
        y_values = np.array(curve_data)
        x_values = np.arange(len(y_values))

        # 绘制曲线
        plt.plot(x_values, y_values,
                color=colors[i],
                linewidth=2,
                label=f'Curve {i+1}',
                marker='o',
                markersize=4)

    # 设置图像属性
    plt.xlabel('Data Point Index')
    plt.ylabel('Value')
    plt.title(f'Curves from {base_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存到: {output_path}")

    # 关闭图形，不显示
    plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='从JSON文件绘制line_data中的多条曲线')
    parser.add_argument('json_file', help='包含line_data的JSON文件路径')

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.json_file):
        print(f"错误: 文件 {args.json_file} 不存在")
        return

    # 绘制曲线
    try:
        plot_curves_from_json(args.json_file)
    except Exception as e:
        print(f"处理文件时出错: {e}")


if __name__ == "__main__":
    main()