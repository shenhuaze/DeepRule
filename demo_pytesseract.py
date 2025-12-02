from PIL import Image
import pytesseract

# 读取图片（把'your_image.jpg'换成你的图片路径）
img = Image.open('demo_ocr.png')

# 识别文字（中文识别记得加lang参数）
text = pytesseract.image_to_string(img, lang='chi_sim+eng') 

# 打印识别结果
print(text)