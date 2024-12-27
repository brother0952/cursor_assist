import serial
import time
import random

# 配置串口
ser = serial.Serial('COM10', 115200)  # 请根据实际情况修改串口号

def generate_adc_data():
    # 模拟6个ADC通道的数据 (0-255范围)
    adc_values = [random.randint(0, 255) for _ in range(6)]
    
    # 构造数据帧
    frame = bytearray()
    frame.append(ord('$'))  # 起始标记 '$'
    frame.extend(adc_values)  # 添加6个ADC值
    frame.append(ord(';'))  # 结束标记 ';'
    
    return frame

try:
    while True:
        # 生成并发送数据
        data = generate_adc_data()
        ser.write(data)
        time.sleep(0.001)  # 1ms延时，与Arduino示例相同
        
except KeyboardInterrupt:
    ser.close()
    print("串口已关闭") 