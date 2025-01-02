import tkinter as tk
from tkinter import filedialog, ttk
import serial
import serial.tools.list_ports
import os
from crypto import Crypto
import threading
import time
from config_manager import ConfigManager

SECRET_KEY = b"MySecretKey12345"  # 16字节密钥
BAUD_RATE = "115200"
CHUNK_SIZE = 32768  # 或 65536，根据实际测试选择最佳值

class FileSenderGUI:
    def __init__(self, use_encryption=False):
        self.root = tk.Tk()
        self.root.title("文件发送器")
        self.root.geometry("400x300")
        
        # 加载配置
        self.config = ConfigManager("sender_config.json")
        
        if use_encryption:
            self.crypto = Crypto(SECRET_KEY)
        else:
            self.crypto = Crypto()
            
        self.setup_gui()
        
    def setup_gui(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建串口设置框架
        port_frame = ttk.LabelFrame(main_frame, text="串口设置", padding="5")
        port_frame.pack(fill=tk.X, pady=2)
        
        # COM口选择和刷新按钮在同一行
        port_row = ttk.Frame(port_frame)
        port_row.pack(fill=tk.X)
        ttk.Label(port_row, text="串口:").pack(side=tk.LEFT, padx=2)
        self.com_var = tk.StringVar(value=self.config.get("port", ""))
        self.com_box = ttk.Combobox(port_row, textvariable=self.com_var, width=15)
        self.refresh_ports()
        self.com_box.pack(side=tk.LEFT, padx=2)
        ttk.Button(port_row, text="刷新", command=self.refresh_ports, width=8).pack(side=tk.LEFT, padx=2)
        
        # 波特率选择
        baud_row = ttk.Frame(port_frame)
        baud_row.pack(fill=tk.X, pady=2)
        ttk.Label(baud_row, text="波特率:").pack(side=tk.LEFT, padx=2)
        self.baud_var = tk.StringVar(value=self.config.get("baudrate", BAUD_RATE))
        self.baud_box = ttk.Combobox(baud_row, textvariable=self.baud_var,
                                    values=["9600", "115200", "256000", "460800", "921600","1000000", "1500000", "2000000", "3000000"],
                                    width=15)
        self.baud_box.pack(side=tk.LEFT, padx=2)
        
        # 添加配置变更监听
        self.com_var.trace_add('write', self.on_port_change)
        self.baud_var.trace_add('write', self.on_baudrate_change)
        
        # 文件选择框架
        file_frame = ttk.LabelFrame(main_frame, text="文件", padding="5")
        file_frame.pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="选择文件", command=self.select_file).pack(side=tk.LEFT, padx=2)
        self.file_label = ttk.Label(file_frame, text="未选择文件")
        self.file_label.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        # 控制按钮框架
        ctrl_frame = ttk.Frame(main_frame)
        ctrl_frame.pack(fill=tk.X, pady=2)
        self.send_button = ttk.Button(ctrl_frame, text="发送", command=self.start_send)
        self.send_button.pack(side=tk.LEFT, padx=2)
        self.stop_button = ttk.Button(ctrl_frame, text="中断", command=self.stop_send,
                                     state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=2)
        
        # 进度框架
        progress_frame = ttk.LabelFrame(main_frame, text="进度", padding="5")
        progress_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        self.progress = ttk.Progressbar(progress_frame, length=300, mode='determinate')
        self.progress.pack(fill=tk.X, pady=2)
        self.status_label = ttk.Label(progress_frame, text="就绪")
        self.status_label.pack(fill=tk.X)
        
        # 添加中断标志
        self.stop_transfer = False
        
    def refresh_ports(self):
        self.com_box['values'] = [port.device for port in serial.tools.list_ports.comports()]
        
    def select_file(self):
        # 使用上次的目录
        initial_dir = self.config.get("last_directory", "")
        filename = filedialog.askopenfilename(initialdir=initial_dir)
        if filename:
            self.filename = filename
            self.file_label.config(text=os.path.basename(filename))
            # 保存当前目录
            self.config.set("last_directory", os.path.dirname(filename))
            
    def send_file(self):
        try:
            print(f"开始发送文件: {self.filename}")
            print(f"串口: {self.com_var.get()}, 波特率: {self.baud_var.get()}")
            
            # 记录开始时间
            start_time = time.time()
            
            with serial.Serial(
                self.com_var.get(), 
                int(self.baud_var.get()), 
                timeout=2,
                write_timeout=2,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False
            ) as ser:
                ser.rts = True
                ser.dtr = True
                time.sleep(0.5)
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                
                # 发送文件名和大小
                filename = os.path.basename(self.filename)
                file_size = os.path.getsize(self.filename)
                print(f"发送文件信息: {filename} ({file_size} bytes)")
                self._send_encrypted_data(ser, filename.encode())
                self._send_encrypted_data(ser, str(file_size).encode())
                
                # 发送文件内容
                sent_size = 0
                with open(self.filename, 'rb') as f:
                    while sent_size < file_size:
                        if self.stop_transfer:
                            raise Exception("用户中断传输")
                            
                        chunk = f.read(CHUNK_SIZE)
                        if not chunk:
                            break
                            
                        print(f"\n发送数据块: {len(chunk)} bytes")
                        self._send_encrypted_data(ser, chunk)
                        
                        # 等待确认
                        print("等待接收端确认...")
                        response = self._receive_encrypted_data(ser)
                        if response != b"OK":
                            raise Exception(f"接收端返回错误: {response}")
                        
                        sent_size += len(chunk)
                        progress = (sent_size / file_size) * 100
                        self.progress['value'] = progress
                        self.status_label.config(text=f"已发送: {sent_size}/{file_size} bytes")
                        self.root.update()
                
                # 计算传输时间和速度
                elapsed_time = time.time() - start_time
                transfer_speed = file_size / (1024 * 1024 * elapsed_time)  # MB/s
                
                print(f"\n文件发送完成")
                print(f"传输时间: {elapsed_time:.2f} 秒")
                print(f"平均速度: {transfer_speed:.2f} MB/s")
                self.status_label.config(text=f"发送成功! 用时: {elapsed_time:.1f}秒, 速度: {transfer_speed:.1f}MB/s")
                
        except Exception as e:
            print(f"发送错误: {str(e)}")
            self.status_label.config(text=f"错误: {str(e)}")
        finally:
            self.send_button.config(state='normal')
            self.stop_button.config(state=tk.DISABLED)
            
    def _send_encrypted_data(self, ser, data):
        try:
            encrypted_data = self.crypto.encrypt(data)
            length_str = f"{len(encrypted_data):08d}".encode()
            print(f"发送数据长度: {length_str}")
            
            # 确保完整发送长度数据
            total_sent = 0
            while total_sent < 8:
                sent = ser.write(length_str[total_sent:])
                if sent == 0:
                    raise Exception("发送长度数据失败")
                total_sent += sent
            
            print(f"发送加密数据: {len(encrypted_data)} bytes")
            # 确保完整发送数据
            total_sent = 0
            while total_sent < len(encrypted_data):
                sent = ser.write(encrypted_data[total_sent:])
                if sent == 0:
                    raise Exception("发送数据失败")
                total_sent += sent
                
            # 等待数据发送完成
            ser.flush()
            
        except Exception as e:
            print(f"发送数据错误: {str(e)}")
            raise
        
    def _receive_encrypted_data(self, ser):
        try:
            # 读取长度数据，确保读取完整的8字节
            length_data = b''
            while len(length_data) < 8:
                data = ser.read(8 - len(length_data))
                if not data:
                    print("读取长度数据超时")
                    raise Exception("读取超时")
                length_data += data
                
            print(f"接收到长度数据: {length_data}")
            length = int(length_data.decode())
            print(f"准备接收数据: {length} bytes")
            
            # 读取数据，确保读取完整的数据
            encrypted_data = b''
            while len(encrypted_data) < length:
                chunk = ser.read(length - len(encrypted_data))
                if not chunk:
                    print("读取数据超时")
                    raise Exception("读取超时")
                encrypted_data += chunk
                
            print(f"接收到加密数据: {len(encrypted_data)} bytes")
            return self.crypto.decrypt(encrypted_data)
        except Exception as e:
            print(f"接收数据错误: {str(e)}")
            raise
        
    def start_send(self):
        if not hasattr(self, 'filename'):
            self.status_label.config(text="请先选择文件")
            return
            
        if not self.com_var.get():
            self.status_label.config(text="请选择串口")
            return
            
        self.progress['value'] = 0
        self.send_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.stop_transfer = False
        
        # 在新线程中发送文件
        self.send_thread = threading.Thread(target=self.send_file)
        self.send_thread.daemon = True
        self.send_thread.start()
        
    def stop_send(self):
        """中断发送"""
        self.stop_transfer = True
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="正在中断发送...")
        
    def run(self):
        self.root.mainloop()
        
    def on_port_change(self, *args):
        """串口变更时保存配置"""
        self.config.set("port", self.com_var.get())
        
    def on_baudrate_change(self, *args):
        """波特率变更时保存配置"""
        self.config.set("baudrate", self.baud_var.get())

def main():
    # 可以通过参数控制是否启用加密
    app = FileSenderGUI(use_encryption=False)
    app.run()

if __name__ == "__main__":
    main() 