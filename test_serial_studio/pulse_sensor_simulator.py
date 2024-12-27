import serial
import time
import math
import random

# 配置串口
ser = serial.Serial('COM10', 115200)

class LowPassFilter:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.value = 0
    
    def update(self, new_value):
        self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

class MovingAverageFilter:
    def __init__(self, size=10):
        self.size = size
        self.buffer = [0] * size
        self.index = 0
        self.sum = 0
        
    def update(self, value):
        self.sum -= self.buffer[self.index]
        self.buffer[self.index] = value
        self.sum += value
        self.index = (self.index + 1) % self.size
        return self.sum / self.size

def generate_pulse_data(t, low_pass, moving_avg):
    # 生成基础心跳信号 (约1Hz，模拟心跳)
    base_signal = math.sin(2 * math.pi * 1.0 * t) * 50 + 512
    
    # 添加高频噪声
    noisy_signal = base_signal + random.uniform(-20, 20)
    
    # 应用低通滤波
    filtered_signal = low_pass.update(noisy_signal)
    
    # 应用移动平均滤波
    final_signal = moving_avg.update(filtered_signal)
    
    # 构造数据帧
    frame = f"{final_signal:.2f}\n"
    return frame.encode()

# 初始化滤波器
low_pass = LowPassFilter(alpha=0.1)
moving_avg = MovingAverageFilter(size=10)
t = 0

try:
    print("脉搏传感器模拟器已启动...")
    while True:
        # 生成并发送数据
        data = generate_pulse_data(t, low_pass, moving_avg)
        ser.write(data)
        
        t += 0.005  # 5ms采样间隔
        time.sleep(0.005)
        
except KeyboardInterrupt:
    ser.close()
    print("\n脉搏传感器模拟器已关闭") 