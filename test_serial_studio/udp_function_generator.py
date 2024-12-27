import socket
import time
import math
import argparse

def generate_wave_value(wave_type, frequency, phase):
    """生成不同类型的波形值"""
    if wave_type == "sine":
        return (math.sin(phase) + 1.0) * 2.5
    elif wave_type == "triangle":
        return abs(((phase / (2 * math.pi)) % 1.0) * 2.0 - 1.0) * 5.0
    elif wave_type == "saw":
        return ((phase / (2 * math.pi)) % 1.0) * 5.0
    elif wave_type == "square":
        return 5.0 if math.sin(phase) >= 0 else 0.0
    return 0.0

def validate_frequency(frequency, interval_ms):
    """验证频率是否合适"""
    nyquist_rate = 1.0 / (2.0 * (interval_ms / 1000.0))
    safe_rate = 0.8 * nyquist_rate
    
    if frequency >= nyquist_rate:
        print(f"警告: 频率 {frequency:.2f} Hz 超过奈奎斯特频率 ({nyquist_rate:.2f} Hz)")
    elif frequency > safe_rate:
        print(f"警告: 频率 {frequency:.2f} Hz 接近奈奎斯特频率，建议降低到 {safe_rate:.2f} Hz 以下")

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='UDP函数发生器')
    parser.add_argument('-p', '--port', type=int, default=9000, help='UDP端口')
    parser.add_argument('-i', '--interval', type=float, default=1.0, help='发送间隔(ms)')
    parser.add_argument('-n', '--num_functions', type=int, default=1, help='波形数量')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细输出')
    args = parser.parse_args()

    # 创建UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # 配置波形
    wave_configs = []
    for i in range(args.num_functions):
        while True:
            wave_type = input(f"波形 {i+1} 类型 (sine/triangle/saw/square): ")
            if wave_type in ["sine", "triangle", "saw", "square"]:
                break
            print("无效的波形类型")
        
        frequency = float(input(f"波形 {i+1} 频率 (Hz): "))
        phase = float(input(f"波形 {i+1} 相位 (弧度): "))
        validate_frequency(frequency, args.interval)
        wave_configs.append((wave_type, frequency, phase))
    
    print("\nUDP函数发生器已启动...")
    try:
        while True:
            values = []
            for i, (wave_type, frequency, phase) in enumerate(wave_configs):
                # 更新相位
                new_phase = phase + 2 * math.pi * frequency * (args.interval / 1000.0)
                wave_configs[i] = (wave_type, frequency, new_phase)
                
                # 生成波形值
                value = generate_wave_value(wave_type, frequency, new_phase)
                values.append(f"{value:.2f}")
            
            # 构造并发送数据帧
            frame = ",".join(values) + "\n"
            sock.sendto(frame.encode(), ('localhost', args.port))
            
            if args.verbose:
                print(f"发送: {frame.strip()}")
            
            time.sleep(args.interval / 1000.0)
            
    except KeyboardInterrupt:
        print("\nUDP函数发生器已关闭")
        sock.close()

if __name__ == "__main__":
    main() 