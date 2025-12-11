#!/usr/bin/env python3
"""
绘图脚本：从JSON文件中读取line_data字段的多条曲线数据并绘制在同一张图中
JSON格式：包含line_data字段，其中每个列表代表一条曲线的数据点
用法:
  - 处理单个文件: python plot_extracted_data.py --json_file <json_file>
  - 批量处理: python plot_extracted_data.py --json_dir <json_dir>
"""

import json
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import glob


def plot_curves_from_json(json_file, output_dir=None):
    """
    从JSON文件读取line_data并绘制多条曲线

    Args:
        json_file (str): JSON文件路径
        output_dir (str, optional): 输出目录。如果为None，则输出到与JSON文件相同目录
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

    # 获取JSON文件的基础名称（不含扩展名）
    base_name = os.path.splitext(os.path.basename(json_file))[0]

    # 确定输出路径
    if output_dir is None:
        # 输出到与JSON文件相同目录
        json_dir = os.path.dirname(json_file)
        if not json_dir:
            json_dir = '.'
        output_path = os.path.join(json_dir, f"{base_name}.png")
    else:
        # 输出到指定目录
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{base_name}.png")

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


def process_single_file(json_file):
    """
    处理单个JSON文件

    Args:
        json_file (str): JSON文件路径
    """
    # 检查文件是否存在
    if not os.path.exists(json_file):
        print(f"错误: 文件 {json_file} 不存在")
        return

    try:
        plot_curves_from_json(json_file)
    except Exception as e:
        print(f"处理文件 {json_file} 时出错: {e}")


def process_directory(json_dir):
    """
    批量处理目录中的所有JSON文件

    Args:
        json_dir (str): 包含JSON文件的目录路径
    """
    # 检查目录是否存在
    if not os.path.exists(json_dir):
        print(f"错误: 目录 '{json_dir}' 不存在")
        return

    # 创建输出目录
    parent_dir = os.path.dirname(os.path.abspath(json_dir))
    output_dir = f"{parent_dir}/{os.path.basename(json_dir)}_plot"

    # 查找所有JSON文件
    json_files = glob.glob(os.path.join(json_dir, "*.json"))

    if not json_files:
        print(f"在目录 {json_dir} 中没有找到JSON文件")
        return

    print(f"找到 {len(json_files)} 个JSON文件")
    print(f"输出目录: {output_dir}")

    # 处理每个文件
    for i, json_file in enumerate(json_files, 1):
        print(f"\n处理第 {i}/{len(json_files)} 个文件: {os.path.basename(json_file)}")
        try:
            plot_curves_from_json(json_file, output_dir)
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {e}")

    print(f"\n批量处理完成！共处理了 {len(json_files)} 个文件")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='从JSON文件绘制line_data中的多条曲线')
    # 创建互斥组：要么指定文件，要么指定目录
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--json_file', help='单个包含line_data的JSON文件路径')
    group.add_argument('--json_dir', help='包含多个JSON文件的目录路径')

    args = parser.parse_args()

    if args.json_file:
        # 处理单个文件
        process_single_file(args.json_file)
    elif args.json_dir:
        # 批量处理目录
        process_directory(args.json_dir)


if __name__ == "__main__":
    main()