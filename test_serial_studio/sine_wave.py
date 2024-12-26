import serial
import math
import time
import json

# 配置串口
ser = serial.Serial('COM10', 256000)  # 请根据实际情况修改串口号

def generate_sine_data(t):
    # 生成一个正弦波和一个余弦波
    sine = math.sin(2 * math.pi * 0.5 * t)
    cosine = math.cos(2 * math.pi * 0.5 * t)
    
    # 构造数据帧 - 使用类似MPU6050的格式
    frame = f"${sine:.6f},{cosine:.6f},{t:.6f};\n"
    return frame.encode()

t = 0
try:
    while True:
        data = generate_sine_data(t)
        ser.write(data)
        t += 0.1
        time.sleep(0.1)  # 100ms发送间隔
        
except KeyboardInterrupt:
    ser.close()
    print("串口已关闭") 