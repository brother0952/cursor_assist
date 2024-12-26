import serial
import math
import time
import json

# 配置串口
ser = serial.Serial('COM10', 256000)  # 请根据实际情况修改串口号

i=0

def generate_sine_data():
    global i
    if i<10:
        i+=1
    else:
        i=0    # 生成一个正弦波和一个余弦波
    # i=str(i)
    res = f"${i:04d}{i:04d}{i:04d}{i:04d}{i:04d}{i:04d};"
    # return str(i).encode()
    return res.encode()


try:
    while True:
        data = generate_sine_data()
        # print(data)
        ser.write(data)
        time.sleep(0.1)  # 100ms发送间隔
        
except KeyboardInterrupt:
    ser.close()
    print("串口已关闭") 