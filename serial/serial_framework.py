import threading
import queue
import serial
import time
from typing import Optional, Any
from protocol_handler import ProtocolHandler, ProtocolMessage

class SerialManager:
    def __init__(self, port: str, protocol_handler: ProtocolHandler, baudrate: int = 115200):
        """
        初始化串口管理器
        :param port: 串口号
        :param protocol_handler: 协议处理器实例
        :param baudrate: 波特率
        """
        self.port = port
        self.baudrate = baudrate
        self.serial: Optional[serial.Serial] = None
        self.protocol_handler = protocol_handler
        
        # 创建数据队列
        self.data_queue = queue.Queue()
        self.message_queue = queue.Queue()  # 用于存放解析后的消息
        
        # 控制线程的标志
        self.is_running = False
        
        # 创建线程
        self.read_thread = threading.Thread(target=self._read_serial_task)
        self.process_thread = threading.Thread(target=self._process_data_task)
        
        # 创建线程锁
        self.serial_lock = threading.Lock()
        
        # 消息处理回调函数字典
        self.message_handlers = {}

    def register_handler(self, message_type: str, handler_func):
        """
        注册消息处理器
        :param message_type: 消息类型
        :param handler_func: 处理函数
        """
        self.message_handlers[message_type] = handler_func

    def start(self):
        """启动串口管理器"""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1
            )
            
            self.is_running = True
            self.read_thread.start()
            self.process_thread.start()
            print(f"串口 {self.port} 已启动")
            
        except serial.SerialException as e:
            print(f"串口打开失败: {e}")
            raise

    def stop(self):
        """停止串口管理器"""
        self.is_running = False
        
        # 等待线程结束
        if self.read_thread.is_alive():
            self.read_thread.join()
        if self.process_thread.is_alive():
            self.process_thread.join()
            
        # 关闭串口
        with self.serial_lock:
            if self.serial and self.serial.is_open:
                self.serial.close()
        print("串口管理器已停止")

    def _read_serial_task(self):
        """串口读取线程"""
        while self.is_running:
            try:
                with self.serial_lock:
                    if self.serial.in_waiting:
                        # 读取串口数据
                        data = self.serial.readline()
                        if data:
                            # 将数据放入队列
                            self.data_queue.put(data)
            except Exception as e:
                print(f"读取串口数据错误: {e}")
                break
            time.sleep(0.001)  # 短暂休眠，避免CPU占用过高

    def _process_data_task(self):
        """数据处理线程"""
        while self.is_running:
            try:
                # 从队列中获取数据
                data = self.data_queue.get(timeout=1)
                
                # 使用协议处理器解析数据
                message = self.protocol_handler.parse_message(data)
                if message:
                    # 如果有对应的处理函数，调用它
                    handler = self.message_handlers.get(message.message_type)
                    if handler:
                        handler(message)
                    else:
                        # 没有处理器，放入消息队列
                        self.message_queue.put(message)
                
                self.data_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理数据错误: {e}")

    def send_message(self, message: Any):
        """
        发送消息
        :param message: 要发送的消息对象
        """
        try:
            # 使用协议处理器构建消息
            data = self.protocol_handler.build_message(message)
            with self.serial_lock:
                self.serial.write(data)
        except Exception as e:
            print(f"发送消息错误: {e}")

# 使用示例
def main():
    from example_protocols import JsonProtocol
    
    # 创建JSON协议处理器
    protocol = JsonProtocol()
    
    # 创建串口管理器实例
    serial_manager = SerialManager(
        port="COM10",
        protocol_handler=protocol
    )
    
    # 定义消息处理函数
    def handle_json_message(message: ProtocolMessage):
        print(f"收到JSON消息: {message.decoded_data}")
    
    # 注册消息处理器
    serial_manager.register_handler("json", handle_json_message)
    
    try:
        # 启动串口管理器
        serial_manager.start()
        
        # 示例：发送一些JSON消息
        for i in range(5):
            message = {
                "command": "test",
                "value": i
            }
            serial_manager.send_message(message)
            time.sleep(1)
        
        # 运行一段时间后停止
        time.sleep(10)
        
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        serial_manager.stop()

if __name__ == "__main__":
    main() 