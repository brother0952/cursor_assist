import serial
import time
import random

# 配置串口
ser = serial.Serial('COM10', 115200)

def generate_lte_data():
    # 模拟LTE信号数据
    cell_id = random.randint(10000, 99999)
    rsrq = random.uniform(-20, 0)      # 参考信号接收质量
    rsrp = random.uniform(-120, -70)   # 参考信号接收功率
    rssi = random.uniform(-100, -50)   # 接收信号强度指示
    sinr = random.uniform(-20, 30)     # 信号干扰噪声比
    
    # 按照LTE modem示例格式构造数据帧
    frame = f"/*{cell_id},{rsrq:.1f},{rsrp:.1f},{rssi:.1f},{sinr:.1f}*/\n"
    return frame.encode()

try:
    print("LTE调制解调器模拟器已启动...")
    while True:
        # 生成并发送数据
        data = generate_lte_data()
        ser.write(data)
        
        # 每5秒更新一次数据
        time.sleep(5)
        
except KeyboardInterrupt:
    ser.close()
    print("\nLTE调制解调器模拟器已关闭") 