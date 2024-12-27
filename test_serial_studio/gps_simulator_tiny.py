import serial
import time
import math

# 配置串口
ser = serial.Serial('COM10', 115200)  # 与Arduino示例使用相同的波特率

def generate_gps_data(t):
    # 模拟一个围绕北京天安门(39.9042° N, 116.4074° E)做圆周运动的GPS轨迹
    # 半径约500米
    radius = 0.005  # 约500米的经纬度变化
    
    # 计算圆周运动的经纬度
    latitude = 39.9042 + radius * math.sin(t)
    longitude = 116.4074 + radius * math.cos(t)
    
    # 模拟高度变化（20-100米之间波动）
    altitude = 60 + 40 * math.sin(t * 0.5)
    
    # 按照TinyGPS示例的格式构造数据帧：$latitude,longitude,altitude;
    frame = f"${latitude:.6f},{longitude:.6f},{altitude:.2f};\n"
    return frame.encode()

t = 0
try:
    print("GPS模拟器已启动...")
    while True:
        # 生成并发送数据
        data = generate_gps_data(t)
        ser.write(data)
        
        # 每次更新角度（控制运动速度）
        t += 0.01
        
        # 与Arduino示例相同的延时
        time.sleep(0.005)  # 5ms延时
        
except KeyboardInterrupt:
    ser.close()
    print("\nGPS模拟器已关闭") 