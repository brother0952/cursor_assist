import serial
import time
import math

# 配置串口
ser = serial.Serial('COM10', 115200)  # 与Arduino示例使用相同的波特率

def generate_lorenz_data(x, y, z, dt):
    # Lorenz系统参数 - 与Arduino示例完全相同
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    
    # 计算导数
    dx = sigma * (y - x) * dt
    dy = (x * (rho - z) - y) * dt
    dz = (x * y - beta * z) * dt
    
    # 更新状态
    x += dx
    y += dy
    z += dz
    
    # 构造数据帧 - 使用与Arduino示例相同的格式
    # 注意：不需要添加开始和结束标记，因为配置文件中使用了换行符作为帧结束
    frame = f"{x:.6f},{y:.6f},{z:.6f}\n"
    return frame.encode(), x, y, z

# 初始条件 - 与Arduino示例相同
x, y, z = 0.1, 0.0, 0.0
dt = 0.01  # 时间步长

# 发送间隔 - 与Arduino示例相同
transmissionInterval = 0.001  # 1ms

try:
    print("Lorenz吸引子模拟器已启动...")
    lastTransmissionTime = time.time()
    
    while True:
        currentTime = time.time()
        if (currentTime - lastTransmissionTime) >= transmissionInterval:
            # 生成并发送数据
            data, x, y, z = generate_lorenz_data(x, y, z, dt)
            ser.write(data)
            lastTransmissionTime = currentTime
            
except KeyboardInterrupt:
    ser.close()
    print("\nLorenz吸引子模拟器已关闭") 