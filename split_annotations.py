import json
import os
import argparse

class CompactEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('indent', 4)
        super().__init__(*args, **kwargs)

    def default(self, obj):
        # 重写default方法来处理bbox数组的格式化
        if isinstance(obj, list) and all(isinstance(x, (int, float)) for x in obj):
            # 对于数值列表（如bbox），返回一个特殊的标记对象
            return BBoxList(obj)
        return super().default(obj)


class BBoxList:
    """包装类用于标记bbox列表，以便特殊处理"""
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def format_json_with_compact_bbox(obj):
    """手动格式化JSON，保持结构但压缩bbox数组"""

    def format_value(value, indent_level=0):
        indent = '    ' * indent_level

        if isinstance(value, dict):
            if not value:
                return '{}'

            items = []
            for k, v in value.items():
                formatted_value = format_value(v, indent_level + 1)
                if k == 'bbox' and isinstance(v, list):
                    # bbox数组特殊处理：压缩为一行
                    bbox_str = '[' + ', '.join(str(x) for x in v) + ']'
                    items.append(f'{indent}"{k}": {bbox_str}')
                else:
                    items.append(f'{indent}"{k}": {formatted_value}')

            return '{\n' + ',\n'.join(items) + '\n' + indent[:-4] + '}'

        elif isinstance(value, list):
            if not value:
                return '[]'

            # 如果是数值列表，可能需要特殊处理
            if all(isinstance(x, (int, float, str)) for x in value) and len(value) <= 20:
                # 短的简单列表，保持在一行
                return '[' + ', '.join(json.dumps(x) for x in value) + ']'

            # 长列表或复杂列表，多行格式化
            items = []
            for item in value:
                formatted_item = format_value(item, indent_level + 1)
                items.append('    ' + indent + formatted_item)

            return '[\n' + ',\n'.join(items) + '\n' + indent + ']'

        else:
            return json.dumps(value, ensure_ascii=False)

    return format_value(obj, 0)

def split_annotations(input_file, output_dir):
    # 读取输入的JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取图片列表和标注列表
    images = data.get('images', [])
    annotations = data.get('annotations', [])
    
    # 遍历每个图片，生成单独的JSON文件
    for image in images:
        image_id = image['id']
        # 筛选当前图片对应的所有标注
        image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]
        
        # 构建输出数据结构
        output_data = {
            "file_name": image['file_name'],
            "image_id": image_id,
            "height": image['height'],
            "width": image['width'],
            "annotations": image_annotations
        }
        
        # 生成输出文件名（将.png替换为.json）
        original_filename = image['file_name']
        output_filename = original_filename.replace('.png', '.json')

        # 构建完整的输出路径
        output_path = os.path.join(output_dir, output_filename)

        # 写入输出文件，使用自定义格式化函数
        with open(output_path, 'w', encoding='utf-8') as f:
            formatted_json = format_json_with_compact_bbox(output_data)
            f.write(formatted_json)

        print(f"已生成文件: {output_path}")

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='将COCO格式的标注文件按图片分割成单独的JSON文件')
    parser.add_argument('input_file', help='输入的JSON标注文件路径')
    parser.add_argument('output_dir', help='输出文件夹路径')

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件 '{args.input_file}' 不存在")
        return

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"创建输出文件夹: {args.output_dir}")

    # 执行分割操作
    split_annotations(args.input_file, args.output_dir)
    print(f"分割完成！文件已保存到: {args.output_dir}")

if __name__ == "__main__":
    main()