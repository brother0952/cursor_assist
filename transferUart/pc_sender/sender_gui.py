import tkinter as tk
from tkinter import filedialog, ttk
import serial
import serial.tools.list_ports
import os
from crypto import Crypto
import threading

SECRET_KEY = b"MySecretKey12345"  # 16字节密钥
BAUD_RATE = 921600
CHUNK_SIZE = 4096

class FileSenderGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("文件发送器")
        self.root.geometry("400x300")
        
        self.crypto = Crypto(SECRET_KEY)
        self.setup_gui()
        
    def setup_gui(self):
        # COM口选择
        tk.Label(self.root, text="串口:").pack(pady=5)
        self.com_var = tk.StringVar()
        self.com_box = ttk.Combobox(self.root, textvariable=self.com_var)
        self.com_box['values'] = [port.device for port in serial.tools.list_ports.comports()]
        self.com_box.pack(pady=5)
        
        # 刷新COM口按钮
        tk.Button(self.root, text="刷新串口", command=self.refresh_ports).pack(pady=5)
        
        # 文件选择
        tk.Button(self.root, text="选择文件", command=self.select_file).pack(pady=10)
        self.file_label = tk.Label(self.root, text="未选择文件")
        self.file_label.pack(pady=5)
        
        # 进度条
        self.progress = ttk.Progressbar(self.root, length=300, mode='determinate')
        self.progress.pack(pady=10)
        
        # 发送按钮
        self.send_button = tk.Button(self.root, text="发送", command=self.start_send)
        self.send_button.pack(pady=10)
        
        # 状态标签
        self.status_label = tk.Label(self.root, text="")
        self.status_label.pack(pady=5)
        
    def refresh_ports(self):
        self.com_box['values'] = [port.device for port in serial.tools.list_ports.comports()]
        
    def select_file(self):
        filename = filedialog.askopenfilename()
        if filename:
            self.filename = filename
            self.file_label.config(text=os.path.basename(filename))
            
    def send_file(self):
        try:
            with serial.Serial(self.com_var.get(), BAUD_RATE, timeout=5) as ser:
                # 发送文件名
                filename = os.path.basename(self.filename)
                self._send_encrypted_data(ser, filename.encode())
                
                # 发送文件大小
                file_size = os.path.getsize(self.filename)
                self._send_encrypted_data(ser, str(file_size).encode())
                
                # 发送文件内容
                sent_size = 0
                with open(self.filename, 'rb') as f:
                    while sent_size < file_size:
                        chunk = f.read(CHUNK_SIZE)
                        if not chunk:
                            break
                            
                        self._send_encrypted_data(ser, chunk)
                        
                        # 等待确认
                        response = self._receive_encrypted_data(ser)
                        if response != b"OK":
                            raise Exception("传输错误")
                            
                        sent_size += len(chunk)
                        progress = (sent_size / file_size) * 100
                        self.progress['value'] = progress
                        self.status_label.config(text=f"已发送: {sent_size}/{file_size} bytes")
                        self.root.update()
                
                self.status_label.config(text="文件发送成功！")
                
        except Exception as e:
            self.status_label.config(text=f"错误: {str(e)}")
            
    def _send_encrypted_data(self, ser, data):
        encrypted_data = self.crypto.encrypt(data)
        ser.write(f"{len(encrypted_data):08d}".encode())
        ser.write(encrypted_data)
        
    def _receive_encrypted_data(self, ser):
        length = int(ser.read(8))
        encrypted_data = ser.read(length)
        return self.crypto.decrypt(encrypted_data)
        
    def start_send(self):
        if not hasattr(self, 'filename'):
            self.status_label.config(text="请先选择文件")
            return
            
        if not self.com_var.get():
            self.status_label.config(text="请选择串口")
            return
            
        self.progress['value'] = 0
        self.send_button.config(state='disabled')
        
        # 在新线程中发送文件
        thread = threading.Thread(target=self.send_file)
        thread.start()
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = FileSenderGUI()
    app.run() 