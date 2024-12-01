import cv2
import numpy as np
import time

def nothing(x):
    pass

# 打开摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 初始化帧率计算的变量
fps = 0
prev_time = time.time()
alpha = 0.1  # 移动平均的权重

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取画面")
        break
    
    # 计算帧率
    current_time = time.time()
    diff = current_time - prev_time
    if diff > 0:
        current_fps = 1 / diff
        fps = fps * (1 - alpha) + current_fps * alpha  # 使用移动平均平滑帧率显示
    prev_time = current_time
    
    # 在画面上显示帧率
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)
    
    cv2.imshow('Camera FPS Demo', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 