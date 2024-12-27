import serial
import time
from datetime import datetime
from pathlib import Path
import queue
import threading
from typing import Optional
import numpy as np

class SerialDataLogger:
    def __init__(self, port: str, baudrate: int = 256000, idle_threshold: float = 0.002):
        """
        初始化串口数据记录器
        
        参数:
            port: 串口号
            baudrate: 波特率
            idle_threshold: 空闲检测阈值（秒），设置更小以提高分帧精度
        """
        self.port = port
        self.baudrate = baudrate
        self.idle_threshold = idle_threshold
        self.serial: Optional[serial.Serial] = None
        
        # 使用numpy数组作为缓冲区，提高性能
        self.buffer_size = baudrate  # 1秒数据量
        self.data_buffer = np.zeros(self.buffer_size, dtype=np.uint8)
        self.buffer_index = 0
        
        # 创建输出目录
        self.output_dir = Path("serial_logs")
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建输出文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self.output_dir / f"serial_log_{timestamp}.txt"
        
        # 控制标志
        self.is_running = False
        self.last_receive_time = 0
        
        # 数据队列
        self.data_queue = queue.Queue(maxsize=1000000)  # 增大队列容量
        
        # 线程
        self.read_thread = None
        self.write_thread = None
        
    def start(self):
        """启动记录器"""
        try:
            # 配置串口
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0  # 非阻塞读取
            )
            
            print(f"串口已打开: {self.port}")
            print(f"数据将保存到: {self.output_file}")
            
            self.is_running = True
            
            # 启动读取和写入线程
            self.read_thread = threading.Thread(target=self._read_task, daemon=True)
            self.write_thread = threading.Thread(target=self._write_task, daemon=True)
            
            self.read_thread.start()
            self.write_thread.start()
            
        except Exception as e:
            print(f"启动失败: {e}")
            self.stop()
            raise
            
    def stop(self):
        """停止记录器"""
        self.is_running = False
        
        if self.read_thread:
            self.read_thread.join(timeout=1)
        if self.write_thread:
            self.write_thread.join(timeout=1)
            
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("串口已关闭")
            
    def _read_task(self):
        """读取串口数据的线程"""
        last_data_time = time.perf_counter()  # 使用高精度计时器
        current_frame = bytearray()
        
        while self.is_running:
            try:
                if self.serial.in_waiting:
                    # 读取可用数据
                    data = self.serial.read(self.serial.in_waiting)
                    current_time = time.perf_counter()
                    
                    # 检查是否需要分帧
                    time_gap = current_time - last_data_time
                    if current_frame and time_gap >= self.idle_threshold:
                        # 将当前帧放入队列
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        self.data_queue.put((timestamp, bytes(current_frame)))
                        current_frame = bytearray()
                    
                    # 添加新数据
                    current_frame.extend(data)
                    last_data_time = current_time
                    
                else:
                    # 使用更短的休眠时间
                    time.sleep(0.0001)  # 100微秒
                    
            except Exception as e:
                print(f"读取错误: {e}")
                break
                
    def _write_task(self):
        """写入数据到文件的线程"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            while self.is_running or not self.data_queue.empty():
                try:
                    # 从队列获取数据
                    timestamp, data = self.data_queue.get(timeout=0.1)
                    
                    # 将数据转换为十六进制字符串
                    hex_data = ' '.join([f"{b:02X}" for b in data])
                    
                    # 写入文件
                    f.write(f"[{timestamp}] {hex_data}\n")
                    f.flush()  # 立即写入磁盘
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"写入错误: {e}")
                    break

def main():
    # 创建记录器实例
    logger = SerialDataLogger(
        port="COM6",
        baudrate=256000,
        idle_threshold=0.002  # 2ms的空闲检测阈值
    )
    
    try:
        logger.start()
        
        # 等待用户按Ctrl+C停止
        print("正在记录数据... 按Ctrl+C停止")
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n停止记录...")
    finally:
        logger.stop()
        print("记录完成")

if __name__ == "__main__":
    main() 