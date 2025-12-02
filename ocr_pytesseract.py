import os
import json
from PIL import Image, ImageEnhance
import pytesseract

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

image_path = './'
for name in ['OCR_temp.png', 'test.png']:
    image_file_path = os.path.join(image_path, name)
    result = ocr_result(image_file_path)
    print(result)
    with open(os.path.join(image_path, name.replace('.png', '.json')), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
