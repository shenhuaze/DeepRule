#!/usr/bin/env python
import os
import json
import torch
import pprint
import argparse
import glob

import matplotlib
matplotlib.use("Agg")
import cv2
from tqdm import tqdm
from config import system_configs
from nnet.py_factory import NetworkFactory
from db.datasets import datasets
import importlib
from RuleGroup.Cls import GroupCls
from RuleGroup.Bar import GroupBar
from RuleGroup.LineQuiry import GroupQuiry
from RuleGroup.LIneMatch import GroupLine
from RuleGroup.Pie import GroupPie
import math
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
torch.backends.cudnn.benchmark = False
import requests
import pytesseract
import time
import re

def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
def load_net(testiter, cfg_name, data_dir, cache_dir, result_dir, cuda_id=0):
    cfg_file = os.path.join(system_configs.config_dir, cfg_name + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)
    configs["system"]["snapshot_name"] = cfg_name
    configs["system"]["data_dir"] = data_dir
    configs["system"]["cache_dir"] = cache_dir
    configs["system"]["result_dir"] = result_dir
    configs["system"]["tar_data_dir"] = "cls"
    system_configs.update_config(configs["system"])

    train_split = system_configs.train_split
    val_split = system_configs.val_split
    test_split = system_configs.test_split

    split = {
        "training": train_split,
        "validation": val_split,
        "testing": test_split
    }["validation"]

    result_dir = system_configs.result_dir
    result_dir = os.path.join(result_dir, str(testiter), split)

    make_dirs([result_dir])

    test_iter = system_configs.max_iter if testiter is None else testiter
    print("loading parameters at iteration: {}".format(test_iter))
    dataset = system_configs.dataset
    db = datasets[dataset](configs["db"], split)
    print("building neural network...")
    nnet = NetworkFactory(db)
    print("loading parameters...")
    nnet.load_params(test_iter)
    if torch.cuda.is_available():
        nnet.cuda(cuda_id)
    nnet.eval_mode()
    return db, nnet

def Pre_load_nets():
    methods = {}
    db_cls, nnet_cls = load_net(50000, "CornerNetCls", "data/clsdata(1031)", "data/clsdata(1031)/cache",
                                "data/clsdata(1031)/result")

    from testfile.test_line_cls_pure_real import testing
    path = 'testfile.test_%s' % "CornerNetCls"
    testing_cls = importlib.import_module(path).testing
    methods['Cls'] = [db_cls, nnet_cls, testing_cls]
    db_bar, nnet_bar = load_net(50000, "CornerNetPureBar", "data/bardata(1031)", "data/bardata(1031)/cache",
                                "data/bardata(1031)/result")
    path = 'testfile.test_%s' % "CornerNetPureBar"
    testing_bar = importlib.import_module(path).testing
    methods['Bar'] = [db_bar, nnet_bar, testing_bar]
    db_pie, nnet_pie = load_net(50000, "CornerNetPurePie", "data/piedata(1008)", "data/piedata(1008)/cache",
                                "data/piedata(1008)/result")
    path = 'testfile.test_%s' % "CornerNetPurePie"
    testing_pie = importlib.import_module(path).testing
    methods['Pie'] = [db_pie, nnet_pie, testing_pie]
    db_line, nnet_line = load_net(50000, "CornerNetLine", "data/linedata(1028)", "data/linedata(1028)/cache",
                                  "data/linedata(1028)/result")
    path = 'testfile.test_%s' % "CornerNetLine"
    testing_line = importlib.import_module(path).testing
    methods['Line'] = [db_line, nnet_line, testing_line]
    db_line_cls, nnet_line_cls = load_net(20000, "CornerNetLineClsReal", "data/linedata(1028)",
                                          "data/linedata(1028)/cache",
                                          "data/linedata(1028)/result")
    path = 'testfile.test_%s' % "CornerNetLineCls"
    testing_line_cls = importlib.import_module(path).testing
    methods['LineCls'] = [db_line_cls, nnet_line_cls, testing_line_cls]
    return methods
methods = Pre_load_nets()

def ocr_result(image_path):
    # 读取和预处理图像
    image = Image.open(image_path)
    enh_con = ImageEnhance.Contrast(image)
    contrast = 2.0
    image = enh_con.enhance(contrast)

    # 使用pytesseract获取详细的OCR数据
    # output_type='dict'返回包含单词信息的字典
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    word_infos = []
    n_boxes = len(ocr_data['level'])

    for i in range(n_boxes):
        # 只处理单词级别（level=5）的结果
        if int(ocr_data['level'][i]) == 5:
            text = ocr_data['text'][i].strip()
            # 跳过空文本
            if not text:
                continue

            # 获取边界框坐标
            left = ocr_data['left'][i]
            top = ocr_data['top'][i]
            width = ocr_data['width'][i]
            height = ocr_data['height'][i]

            # 计算4个角点的坐标 [x1, y1, x2, y2, x3, y3, x4, y4]
            # 顺序：左上角、右上角、右下角、左下角
            bounding_box = [
                left, top,                    # 左上角
                left + width, top,            # 右上角
                left + width, top + height,   # 右下角
                left, top + height            # 左下角
            ]

            # 构建与原始格式一致的单词信息字典
            word_info = {
                'boundingBox': bounding_box,
                'text': text
            }

            # 如果存在置信度，添加到字典中
            if 'conf' in ocr_data and ocr_data['conf'][i] != '-1':
                confidence = float(ocr_data['conf'][i])
                if confidence < 50:  # 过滤低置信度的结果
                    continue
                word_info['confidence'] = confidence

            word_infos.append(word_info)

    return word_infos

def check_intersection(box1, box2):
    if (box1[2] - box1[0]) + ((box2[2] - box2[0])) > max(box2[2], box1[2]) - min(box2[0], box1[0]) \
            and (box1[3] - box1[1]) + ((box2[3] - box2[1])) > max(box2[3], box1[3]) - min(box2[1], box1[1]):
        Xc1 = max(box1[0], box2[0])
        Yc1 = max(box1[1], box2[1])
        Xc2 = min(box1[2], box2[2])
        Yc2 = min(box1[3], box2[3])
        intersection_area = (Xc2-Xc1)*(Yc2-Yc1)
        return intersection_area/((box2[3]-box2[1])*(box2[2]-box2[0]))
    else:
        return 0

def try_math(image_path, cls_info):
    title_list = [1, 2, 3]
    title2string = {}
    max_value = 1
    min_value = 0
    max_y = 0
    min_y = 1
    word_infos = ocr_result(image_path)
    print("ocr_result:", word_infos)
    for id in title_list:
        if id in cls_info.keys():
            predicted_box = cls_info[id]
            words = []
            for word_info in word_infos:
                word_bbox = [word_info["boundingBox"][0], word_info["boundingBox"][1], word_info["boundingBox"][4], word_info["boundingBox"][5]]
                if check_intersection(predicted_box, word_bbox) > 0.5:
                    words.append([word_info["text"], word_bbox[0], word_bbox[1]])
            words.sort(key=lambda x: x[1]+10*x[2])
            word_string = ""
            for word in words:
                word_string = word_string + word[0] + ' '
            title2string[id] = word_string
    if 5 in cls_info.keys():
        plot_area = cls_info[5]
        y_max = plot_area[1]
        y_min = plot_area[3]
        x_board = plot_area[0]
        dis_max = 10000000000000000
        dis_min = 10000000000000000
        for word_info in word_infos:
            word_bbox = [word_info["boundingBox"][0], word_info["boundingBox"][1], word_info["boundingBox"][4], word_info["boundingBox"][5]]
            word_text = word_info["text"]
            word_text = re.sub('[^-+0123456789.]', '',  word_text)
            word_text_num = re.sub('[^0123456789]', '', word_text)
            word_text_pure = re.sub('[^0123456789.]', '', word_text)
            if len(word_text_num) > 0 and word_bbox[2] <= x_board+10:
                dis2max = math.sqrt(math.pow((word_bbox[0]+word_bbox[2])/2-x_board, 2)+math.pow((word_bbox[1]+word_bbox[3])/2-y_max, 2))
                dis2min = math.sqrt(math.pow((word_bbox[0] + word_bbox[2]) / 2 - x_board, 2) + math.pow(
                    (word_bbox[1] + word_bbox[3]) / 2 - y_min, 2))
                y_mid = (word_bbox[1]+word_bbox[3])/2
                if dis2max <= dis_max:
                    dis_max = dis2max
                    max_y = y_mid
                    max_value = float(word_text_pure)
                    if word_text[0] == '-':
                        max_value = -max_value
                if dis2min <= dis_min:
                    dis_min = dis2min
                    min_y = y_mid
                    min_value = float(word_text_pure)
                    if word_text[0] == '-':
                        min_value = -min_value
        print(min_value)
        print(max_value)
        delta_min_max = max_value-min_value
        delta_mark = min_y - max_y
        delta_plot_y = y_min - y_max
        # 防止除零错误
        if delta_mark == 0:
            print("Warning: delta_mark is zero, using default scaling")
            delta = 1.0
        else:
            delta = delta_min_max/delta_mark
        # 防止除零错误
        if delta_plot_y != 0 and abs(min_y-y_min)/delta_plot_y > 0.1:
            print(abs(min_y-y_min)/delta_plot_y)
            print("Predict the lower bar")
            min_value = int(min_value + (min_y-y_min)*delta)

    return title2string, round(min_value, 2), round(max_value, 2)


def test(image_path, debug=False, suffix=None, min_value_official=None, max_value_official=None):
    image_cls = Image.open(image_path)
    image = cv2.imread(image_path)
    json_info = {}
    with torch.no_grad():
        results = methods['Cls'][2](image, methods['Cls'][0], methods['Cls'][1], debug=False)
        info = results[0]
        tls = results[1]
        brs = results[2]
        plot_area = []
        image_painted, cls_info = GroupCls(image_cls, tls, brs)
        title2string, min_value, max_value = try_math(image_path, cls_info)
        
        if min_value_official is not None:
            min_value = min_value_official
            max_value = max_value_official
        chartinfo = {
            "data_type": info["data_type"],
            "cls_info": cls_info,
            "title2string": title2string,
            "min_value": min_value,
            "max_value": max_value
            }
        
        if info['data_type'] == 0:
            print("Predicted as BarChart")
            results = methods['Bar'][2](image, methods['Bar'][0], methods['Bar'][1], debug=False)
            tls = results[0]
            brs = results[1]
            if 5 in cls_info.keys():
                plot_area = cls_info[5][0:4]
            else:
                plot_area = [0, 0, 600, 400]
            image_painted, bar_data = GroupBar(image_painted, tls, brs, plot_area, min_value, max_value)
            json_info["plot_area"] = plot_area
            json_info["bar_data"] = bar_data
            json_info["chartinfo"] = chartinfo
        
        if info['data_type'] == 2:
            print("Predicted as PieChart")
            results = methods['Pie'][2](image, methods['Pie'][0], methods['Pie'][1], debug=False)
            cens = results[0]
            keys = results[1]
            image_painted, pie_data = GroupPie(image_painted, cens, keys)
            json_info["plot_area"] = plot_area
            json_info["pie_data"] = pie_data
            json_info["chartinfo"] = chartinfo
        
        if info['data_type'] == 1:
            print("Predicted as LineChart")
            results = methods['Line'][2](image, methods['Line'][0], methods['Line'][1], debug=False, cuda_id=0)
            keys = results[0]
            hybrids = results[1]
            if 5 in cls_info.keys():
                plot_area = cls_info[5][0:4]
            else:
                plot_area = [0, 0, 600, 400]
            print(min_value, max_value)
            image_painted, quiry, keys, hybrids = GroupQuiry(image_painted, keys, hybrids, plot_area, min_value, max_value)
            results = methods['LineCls'][2](image, methods['LineCls'][0], quiry, methods['LineCls'][1], debug=False, cuda_id=0)
            line_data = GroupLine(image_painted, keys, hybrids, plot_area, results, min_value, max_value)
            json_info["plot_area"] = plot_area
            json_info["line_data"] = line_data
            json_info["chartinfo"] = chartinfo

    return json_info


def parse_args():
    parser = argparse.ArgumentParser(description="Test DeepRule")
    parser.add_argument("--image_file", dest="image_file", help="single image file to process", default=None, type=str)
    parser.add_argument("--image_dir", dest="image_dir", help="directory containing images to process", default=None, type=str)
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    args = parser.parse_args()

    # 检查参数
    if args.image_file is None and args.image_dir is None:
        raise ValueError("必须指定 --image_file 或 --image_dir 参数")
    if args.image_file is not None and args.image_dir is not None:
        raise ValueError("不能同时指定 --image_file 和 --image_dir 参数")

    return args


def get_image_files(directory):
    """获取目录中的所有图片文件"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []

    for ext in image_extensions:
        # 支持大小写
        image_files.extend(glob.glob(os.path.join(directory, f'*{ext}')))
        image_files.extend(glob.glob(os.path.join(directory, f'*{ext.upper()}')))

    return sorted(image_files)


def process_single_image(image_path, debug=False, output_dir=None):
    """处理单张图片"""
    try:
        print(f"正在处理图片: {image_path}")
        json_info_ = test(image_path, debug=debug)

        # 生成输出文件路径
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        if output_dir:
            # 批量处理模式：输出到指定目录
            make_dirs([output_dir])
            output_path = os.path.join(output_dir, f"{image_name}_extracted.json")
        else:
            # 单张图片模式：输出到原图所在目录
            output_path = os.path.splitext(image_path)[0] + "_extracted.json"

        # 保存JSON文件
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(json_info_, f, ensure_ascii=False, indent=4)

        print(f"处理完成，结果已保存到: {output_path}")
        return True

    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {str(e)}")
        return False


if __name__ == "__main__":
    args = parse_args()

    if args.image_file:
        # 单张图片模式
        print(f"单张图片模式: {args.image_file}")
        process_single_image(args.image_file, debug=args.debug)

    elif args.image_dir:
        # 批量处理模式
        print(f"批量处理模式: {args.image_dir}")

        if not os.path.exists(args.image_dir):
            print(f"错误: 目录 {args.image_dir} 不存在")
            exit(1)

        # 创建输出目录
        # 移除路径末尾的斜杠
        clean_path = args.image_dir.rstrip(os.sep)
        parent_dir = os.path.dirname(clean_path)
        dir_name = os.path.basename(clean_path)
        output_dir = os.path.join(parent_dir, dir_name + "_extracted")

        # 获取所有图片文件
        image_files = get_image_files(args.image_dir)

        if not image_files:
            print(f"在目录 {args.image_dir} 中未找到图片文件")
            exit(1)

        print(f"找到 {len(image_files)} 张图片")
        print(f"输出目录: {output_dir}")

        # 批量处理
        success_count = 0
        for i, image_path in enumerate(image_files, 1):
            print(f"\n进度: {i}/{len(image_files)}")
            if process_single_image(image_path, debug=args.debug, output_dir=output_dir):
                success_count += 1

        print(f"\n批量处理完成! 成功处理 {success_count}/{len(image_files)} 张图片")
        print(f"所有结果已保存到: {output_dir}")
