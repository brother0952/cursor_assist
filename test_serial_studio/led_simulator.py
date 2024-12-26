import serial
import time
import json

# 配置串口
ser = serial.Serial('COM10', 256000)  # 请根据实际情况修改串口号

def generate_led_data(counter):
    # 模拟3个LED的状态
    led1 = counter % 2  # 每次切换
    led2 = (counter // 2) % 2  # 每2次切换
    led3 = (counter // 4) % 2  # 每4次切换
    brightness = (counter % 100) / 100.0
    temperature = 25 + (counter % 10)
    
    # 构造数据帧 - 使用类似LTE modem的格式
    frame = f"/*{led1},{led2},{led3},{brightness:.2f},{temperature}*/\n"
    return frame.encode()

counter = 0
try:
    while True:
        data = generate_led_data(counter)
        ser.write(data)
        counter += 1
        time.sleep(0.5)  # 500ms更新一次
        
except KeyboardInterrupt:
    ser.close()
    print("串口已关闭") 