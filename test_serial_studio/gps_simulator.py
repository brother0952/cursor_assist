import serial
import time
import json
import random

# 配置串口
ser = serial.Serial('COM10', 256000)  # 请根据实际情况修改串口号

def generate_gps_data():
    # 模拟一个在北京附近移动的GPS
    lat = 39.9042 + random.uniform(-0.01, 0.01)
    lon = 116.4074 + random.uniform(-0.01, 0.01)
    alt = 50 + random.uniform(-5, 5)
    satellites = random.randint(8, 12)
    speed = random.uniform(0, 50)
    
    # 构造数据帧 - 使用类似LorenzAttractor的格式
    frame = f"{lat:.6f},{lon:.6f},{alt:.1f},{satellites},{speed:.1f}\n"
    return frame.encode()

try:
    while True:
        data = generate_gps_data()
        ser.write(data)
        time.sleep(1)  # 每秒更新一次位置
        
except KeyboardInterrupt:
    ser.close()
    print("串口已关闭") 