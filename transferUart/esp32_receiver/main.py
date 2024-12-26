from machine import UART, Pin
import os
from crypto import Crypto
import gc

# 配置参数
BAUD_RATE = 921600  # 最大波特率
BUFFER_SIZE = 4096
SECRET_KEY = b"MySecretKey12345"  # 16字节密钥

class FileReceiver:
    def __init__(self):
        # 初始化串口
        self.uart = UART(0, baudrate=BAUD_RATE)
        self.uart.init(baudrate=BAUD_RATE, bits=8, parity=None, stop=1)
        self.crypto = Crypto(SECRET_KEY)
        
    def receive_file(self):
        try:
            # 等待文件名
            filename = self._receive_encrypted_data().decode('utf-8')
            print(f"Receiving file: {filename}")
            
            # 等待文件大小
            file_size = int(self._receive_encrypted_data().decode('utf-8'))
            print(f"File size: {file_size} bytes")
            
            # 创建文件
            with open(f"/sd/{filename}", "wb") as f:
                received_size = 0
                while received_size < file_size:
                    # 接收加密数据块
                    data = self._receive_encrypted_data()
                    if not data:
                        raise Exception("Connection lost")
                    
                    f.write(data)
                    received_size += len(data)
                    print(f"Progress: {received_size}/{file_size} bytes")
                    
                    # 发送确认
                    self._send_encrypted_data(b"OK")
                    
                    # 垃圾回收
                    gc.collect()
            
            print("File received successfully")
            return True
            
        except Exception as e:
            print(f"Error receiving file: {str(e)}")
            return False
    
    def _receive_encrypted_data(self):
        # 接收数据长度
        length = int(self.uart.read(8))
        # 接收加密数据
        encrypted_data = self.uart.read(length)
        # 解密数据
        return self.crypto.decrypt(encrypted_data)
    
    def _send_encrypted_data(self, data):
        # 加密数据
        encrypted_data = self.crypto.encrypt(data)
        # 发送数据长度
        self.uart.write(f"{len(encrypted_data):08d}".encode())
        # 发送加密数据
        self.uart.write(encrypted_data)

# 主循环
def main():
    receiver = FileReceiver()
    while True:
        receiver.receive_file()

if __name__ == "__main__":
    main() 