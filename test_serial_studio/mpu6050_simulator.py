import serial
import time
import math

# 配置串口
ser = serial.Serial('COM10', 115200)  # 与Arduino示例使用相同的波特率

def generate_motion_data(t):
    # 模拟运动数据
    # 1. 加速度计数据 (m/s²)
    # 模拟一个简单的圆周运动
    accel_x = 9.81 * math.sin(t)  # 模拟X轴加速度
    accel_y = 9.81 * math.cos(t)  # 模拟Y轴加速度
    accel_z = 9.81  # 保持垂直方向的重力加速度
    
    # 2. 陀螺仪数据 (deg/s)
    # 模拟旋转运动
    gyro_x = 45 * math.sin(t * 2)  # 模拟绕X轴旋转
    gyro_y = 45 * math.cos(t * 2)  # 模拟绕Y轴旋转
    gyro_z = 30 * math.sin(t)      # 模拟绕Z轴旋转
    
    # 3. 温度数据 (°C)
    # 模拟温度在20-30度之间波动
    temperature = 25 + 5 * math.sin(t * 0.1)
    
    # 按照MPU6050示例的格式构造数据帧：
    # $accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,temperature;
    frame = f"${accel_x:.2f},{accel_y:.2f},{accel_z:.2f}," \
           f"{gyro_x:.2f},{gyro_y:.2f},{gyro_z:.2f}," \
           f"{temperature:.2f};\n"
    
    return frame.encode()

t = 0
try:
    print("MPU6050模拟器已启动...")
    while True:
        # 生成并发送数据
        data = generate_motion_data(t)
        ser.write(data)
        
        # 更新时间
        t += 0.01
        
        # 与Arduino示例相同的延时
        time.sleep(0.01)  # 10ms延时
        
except KeyboardInterrupt:
    ser.close()
    print("\nMPU6050模拟器已关闭") 