import cv2
import numpy as np

def nothing(x):
    pass

# 打开摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 打印所有可用的摄像头参数
print("摄像头参数信息：")
for i in range(0, 100):
    value = cap.get(i)
    if value != -1:
        print(f"参数 {i}: {value}")

# 尝试使用不同的方式设置曝光
print("\n尝试设置曝光：")
# 方式1：使用CAP_PROP_EXPOSURE
result1 = cap.set(cv2.CAP_PROP_EXPOSURE, -5)
print(f"方式1设置结果: {result1}, 当前值: {cap.get(cv2.CAP_PROP_EXPOSURE)}")

# 方式2：直接使用ID 15
result2 = cap.set(15, -5)
print(f"方式2设置结果: {result2}, 当前值: {cap.get(15)}")

# 创建窗口和滑动条
cv2.namedWindow('Camera Controls')
cv2.createTrackbar('Exposure', 'Camera Controls', 50, 100, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取画面")
        break
    
    exposure = cv2.getTrackbarPos('Exposure', 'Camera Controls')
    
    # 尝试多种曝光设置方式
    exposure_value = -exposure
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 禁用自动曝光
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
    cap.set(15, exposure_value)  # 直接使用ID
    
    # 显示所有相关值
    actual_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    actual_exposure2 = cap.get(15)
    auto_exposure = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    
    info_text = [
        f'Set: {exposure_value}',
        f'Get(EXPOSURE): {actual_exposure}',
        f'Get(15): {actual_exposure2}',
        f'Auto: {auto_exposure}'
    ]
    
    # 显示多行信息
    for i, text in enumerate(info_text):
        cv2.putText(frame, text, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
    
    cv2.imshow('Camera', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 