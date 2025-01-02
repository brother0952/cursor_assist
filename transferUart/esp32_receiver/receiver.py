import machine
import os
from crypto import Crypto

class FileReceiver:
    def __init__(self, uart_id=2, tx_pin=17, rx_pin=16, baudrate=921600, use_encryption=False):
        # 初始化UART
        self.uart = machine.UART(uart_id, baudrate)
        self.uart.init(baudrate=baudrate, 
                      tx=machine.Pin(tx_pin), 
                      rx=machine.Pin(rx_pin),
                      timeout=1000)  # 超时时间1秒
        
        # 初始化加密模块
        if use_encryption:
            self.crypto = Crypto(b"MySecretKey12345")
        else:
            self.crypto = Crypto()
            
        # 确保存储目录存在
        self.save_dir = "/sd"  # SD卡根目录
        try:
            os.mkdir(self.save_dir)
        except:
            pass
            
    def _receive_data(self):
        """接收数据块"""
        try:
            # 读取长度数据（8字节）
            length_data = b''
            while len(length_data) < 8:
                chunk = self.uart.read(8 - len(length_data))
                if not chunk:
                    raise Exception("读取长度数据超时")
                length_data += chunk
            
            # 解析长度
            length = int(length_data.decode())
            print(f"准备接收数据: {length} bytes")
            
            # 读取数据
            data = b''
            while len(data) < length:
                chunk = self.uart.read(min(1024, length - len(data)))  # 每次最多读取1KB
                if not chunk:
                    raise Exception("读取数据超时")
                data += chunk
            
            # 解密数据
            return self.crypto.decrypt(data)
            
        except Exception as e:
            print(f"接收数据错误: {str(e)}")
            raise
            
    def _send_data(self, data):
        """发送数据"""
        try:
            encrypted_data = self.crypto.encrypt(data)
            length_str = f"{len(encrypted_data):08d}".encode()
            
            # 发送长度
            self.uart.write(length_str)
            
            # 发送数据
            self.uart.write(encrypted_data)
            
        except Exception as e:
            print(f"发送数据错误: {str(e)}")
            raise
            
    def receive_file(self):
        """接收文件"""
        try:
            print("\n等待接收文件...")
            
            # 接收文件名
            filename = self._receive_data().decode('utf-8')
            print(f"接收到文件名: {filename}")
            
            # 发送确认
            self._send_data(b"OK")
            
            # 接收文件大小
            file_size_data = self._receive_data().decode('utf-8')
            file_size = int(file_size_data)
            print(f"文件大小: {file_size} bytes")
            
            # 发送确认
            self._send_data(b"OK")
            
            # 生成保存路径
            save_path = f"{self.save_dir}/{filename}"
            print(f"保存路径: {save_path}")
            
            # 接收文件内容
            received_size = 0
            with open(save_path, 'wb') as f:
                while received_size < file_size:
                    try:
                        data = self._receive_data()
                        if not data:
                            raise Exception("接收到空数据")
                        
                        # 写入文件
                        f.write(data)
                        received_size += len(data)
                        
                        # 发送确认
                        self._send_data(b"OK")
                        
                        # 显示进度
                        print(f"进度: {received_size}/{file_size} bytes")
                        
                    except Exception as e:
                        print(f"接收数据块错误: {str(e)}")
                        try:
                            self._send_data(f"ERROR: {str(e)}".encode())
                        except:
                            pass
                        raise
            
            print("文件接收完成")
            
        except Exception as e:
            print(f"接收错误: {str(e)}")
            raise
            
    def run(self):
        """运行接收器"""
        print("启动文件接收器...")
        print(f"串口配置: UART{self.uart.id}, 波特率: {self.uart.baudrate}")
        
        while True:
            try:
                self.receive_file()
            except Exception as e:
                print(f"等待新的文件...")
                continue

def main():
    # 创建接收器实例
    receiver = FileReceiver(uart_id=2,          # UART2
                          tx_pin=17,            # TX引脚
                          rx_pin=16,            # RX引脚
                          baudrate=921600,      # 波特率
                          use_encryption=False)  # 是否使用加密
    
    # 运行接收器
    receiver.run()

if __name__ == "__main__":
    main() 