# -*-coding:utf-8 -*-


import cv2
import numpy as np
from torchvision import datasets

# 加载MNIST数据集
mnist_dataset = datasets.MNIST('./data', train=True, download=True)

# 获取一张图片和标签
# image, label = mnist_dataset[0]  # 获取第一张图片
image, label = mnist_dataset[20]  # 获取第20张图片

# 将图像从PIL.Image转换为NumPy数组
image = np.array(image)  # 直接将PIL.Image转换为NumPy数组

# 将图像的像素值从[0, 1]范围转换为[0, 255]范围
image = (image * 255).astype('uint8')  # 如果是浮点数格式，进行转换

# 调整图像大小到400x400
image_resized = cv2.resize(image, (400, 400))

# 使用OpenCV显示图像
cv2.imshow(f'Label: {label}', image_resized)
cv2.waitKey(0)  # 等待按键
cv2.destroyAllWindows()  # 关闭窗口