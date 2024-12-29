import serial
import time
from datetime import datetime
from pathlib import Path
import queue
import threading
from typing import Optional
from collections import deque

class SerialDataLogger:
    def __init__(self, port: str, baudrate: int = 500000, idle_threshold: float = 0.003):
        """
        初始化串口数据记录器
        
        参数:
            port: 串口号
            baudrate: 波特率
            idle_threshold: 空闲检测阈值（秒）
        """
        self.port = port
        self.baudrate = baudrate
        self.idle_threshold = idle_threshold
        self.serial: Optional[serial.Serial] = None
        
        # 优化缓冲区设计
        self.read_buffer = memoryview(bytearray(4096)).cast('B')  # 固定大小的读取缓冲区
        self.process_buffer = bytearray(8192)  # 处理缓冲区
        self.buffer_pos = 0
        self.frame_buffer = deque(maxlen=10000)  # 增大帧缓冲队列
        
        # 创建输出目录
        self.output_dir = Path("serial_logs")
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建输出文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self.output_dir / f"serial_log_{timestamp}.txt"
        
        # 控制标志
        self.is_running = False
        
        # 数据队列和事件
        self.data_queue = queue.Queue(maxsize=100000)
        self.write_event = threading.Event()
        
        # 线程
        self.read_thread = None
        self.process_thread = None
        self.write_thread = None
        
        # 性能统计
        self.total_bytes = 0
        self.start_time = 0
        
    def start(self):
        """启动记录器"""
        try:
            # 配置串口
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0,  # 非阻塞读取
                write_timeout=0,  # 非阻塞写入
                exclusive=True  # 独占模式
            )
            
            # 清空缓冲区
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            
            print(f"串口已打开: {self.port}")
            print(f"数据将保存到: {self.output_file}")
            
            self.is_running = True
            self.start_time = time.perf_counter_ns()
            
            # 启动线程
            self.read_thread = threading.Thread(target=self._read_task, daemon=True)
            self.process_thread = threading.Thread(target=self._process_task, daemon=True)
            self.write_thread = threading.Thread(target=self._write_task, daemon=True)
            
            self.read_thread.start()
            self.process_thread.start()
            self.write_thread.start()
            
        except Exception as e:
            print(f"启动失败: {e}")
            self.stop()
            raise
            
    def stop(self):
        """停止记录器"""
        self.is_running = False
        
        # 等待线程结束
        for thread in [self.read_thread, self.process_thread, self.write_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=1)
        
        # 关闭串口
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("串口已关闭")
            
        # 打印统计信息
        duration = (time.perf_counter_ns() - self.start_time) / 1e9
        if duration > 0:
            print(f"\n统计信息:")
            print(f"总接收字节数: {self.total_bytes}")
            print(f"平均速率: {self.total_bytes/duration:.2f} 字节/秒")
            
    def _read_task(self):
        """读取串口数据的线程"""
        while self.is_running:
            try:
                if self.serial.in_waiting:
                    # 使用固定大小读取，避免频繁内存分配
                    bytes_read = self.serial.readinto(self.read_buffer)
                    if bytes_read > 0:
                        timestamp = time.perf_counter_ns()
                        # 只复制实际读取的数据
                        self.frame_buffer.append((timestamp, bytes(self.read_buffer[:bytes_read])))
                        self.total_bytes += bytes_read
                else:
                    time.sleep(0.00005)  # 50微秒的休眠
                    
            except Exception as e:
                print(f"读取错误: {e}")
                break
                
    def _process_task(self):
        """处理数据的线程"""
        last_time = time.perf_counter_ns()
        buffer_pos = 0
        
        while self.is_running or len(self.frame_buffer) > 0:
            try:
                while self.frame_buffer:
                    timestamp, data = self.frame_buffer.popleft()
                    time_gap_ns = timestamp - last_time
                    
                    # 使用更精确的帧分割逻辑
                    if buffer_pos > 0 and time_gap_ns >= self.idle_threshold * 1_000_000_000:
                        frame_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        self.data_queue.put((frame_timestamp, bytes(self.process_buffer[:buffer_pos])))
                        buffer_pos = 0
                    
                    # 复制数据到处理缓冲区
                    self.process_buffer[buffer_pos:buffer_pos + len(data)] = data
                    buffer_pos += len(data)
                    last_time = timestamp
                
                if self.is_running:
                    time.sleep(0.00005)  # 50微秒的休眠
                    
            except Exception as e:
                print(f"处理错误: {e}")
                break
                
    def _write_task(self):
        """写入数据到文件的线程"""
        with open(self.output_file, 'w', encoding='utf-8', buffering=65536) as f:
            write_buffer = []
            last_flush = time.perf_counter()
            
            while self.is_running or not self.data_queue.empty():
                try:
                    timestamp, data = self.data_queue.get(timeout=0.1)
                    hex_data = ' '.join([f"{b:02X}" for b in data])
                    write_buffer.append(f"[{timestamp}] {hex_data}\n")
                    
                    # 批量写入或定时刷新
                    if len(write_buffer) >= 100 or (time.perf_counter() - last_flush) > 0.1:
                        f.writelines(write_buffer)
                        f.flush()
                        write_buffer.clear()
                        last_flush = time.perf_counter()
                        
                except queue.Empty:
                    if write_buffer:  # 清空剩余数据
                        f.writelines(write_buffer)
                        f.flush()
                        write_buffer.clear()
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