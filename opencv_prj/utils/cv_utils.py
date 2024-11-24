import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import platform

def put_chinese_text(img, text, position, font_size=32, color=(0, 255, 0)):
    """在OpenCV图像上显示中文"""
    # 判断操作系统，使用不同的默认字体
    if platform.system() == 'Windows':
        font_path = "C:/Windows/Fonts/simhei.ttf"
    elif platform.system() == 'Linux':
        font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    else:  # macOS
        font_path = "/System/Library/Fonts/PingFang.ttc"
    
    # 创建PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 加载字体
    font = ImageFont.truetype(font_path, font_size)
    
    # 绘制文字
    draw.text(position, text, font=font, fill=color[::-1])  # RGB顺序转换
    
    # 转换回OpenCV格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR) 