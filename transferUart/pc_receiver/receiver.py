import serial
import serial.tools.list_ports
import os
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from crypto import Crypto
import threading
import time

class FileReceiver:
    def __init__(self, use_encryption=False):
        self.root = tk.Tk()
        self.root.title("文件接收器")
        self.root.geometry("400x300")
        
        if use_encryption:
            self.crypto = Crypto(b"MySecretKey12345")
        else:
            self.crypto = Crypto()
            
        self.is_receiving = False
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
        self.com_var = tk.StringVar()
        self.com_box = ttk.Combobox(port_row, textvariable=self.com_var, width=15)
        self.refresh_ports()
        self.com_box.pack(side=tk.LEFT, padx=2)
        ttk.Button(port_row, text="刷新", command=self.refresh_ports, width=8).pack(side=tk.LEFT, padx=2)
        
        # 波特率选择
        baud_row = ttk.Frame(port_frame)
        baud_row.pack(fill=tk.X, pady=2)
        ttk.Label(baud_row, text="波特率:").pack(side=tk.LEFT, padx=2)
        self.baud_var = tk.StringVar(value="115200")
        self.baud_box = ttk.Combobox(baud_row, textvariable=self.baud_var, 
                                    values=["9600", "115200", "256000", "460800", "921600"],
                                    width=15)
        self.baud_box.pack(side=tk.LEFT, padx=2)
        
        # 控制按钮框架
        ctrl_frame = ttk.Frame(main_frame)
        ctrl_frame.pack(fill=tk.X, pady=2)
        self.control_button = ttk.Button(ctrl_frame, text="启动接收", command=self.toggle_receiving)
        self.control_button.pack(side=tk.LEFT, padx=2)
        self.stop_current_button = ttk.Button(ctrl_frame, text="中断当前", command=self.stop_current,
                                            state=tk.DISABLED)
        self.stop_current_button.pack(side=tk.LEFT, padx=2)
        
        # 状态显示框架
        status_frame = ttk.LabelFrame(main_frame, text="状态", padding="5")
        status_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        
        # 保存目录显示
        self.save_dir = os.path.join(os.getcwd(), "received_files")
        os.makedirs(self.save_dir, exist_ok=True)
        ttk.Label(status_frame, text=f"保存目录: {self.save_dir}").pack(fill=tk.X)
        
        # 进度条
        self.progress = ttk.Progressbar(status_frame, length=300, mode='determinate')
        self.progress.pack(fill=tk.X, pady=2)
        
        # 状态标签
        self.status_label = ttk.Label(status_frame, text="就绪")
        self.status_label.pack(fill=tk.X)
        
        # 添加中断标志
        self.stop_current_transfer = False

    def refresh_ports(self):
        self.com_box['values'] = [port.device for port in serial.tools.list_ports.comports()]
        
    def toggle_receiving(self):
        if not self.is_receiving:
            if not self.com_var.get():
                self.status_label.config(text="请选择串口")
                return
                
            self.is_receiving = True
            self.control_button.config(text="停止接收")
            self.com_box.config(state='disabled')
            self.baud_box.config(state='disabled')
            
            # 启动接收线程
            self.receive_thread = threading.Thread(target=self.continuous_receive)
            self.receive_thread.daemon = True
            self.receive_thread.start()
        else:
            self.is_receiving = False
            self.control_button.config(text="启动接收")
            self.com_box.config(state='normal')
            self.baud_box.config(state='normal')
            
    def continuous_receive(self):
        try:
            print(f"\n开始监听串口 {self.com_var.get()} (波特率: {self.baud_var.get()})")
            
            with serial.Serial(
                self.com_var.get(), 
                int(self.baud_var.get()), 
                timeout=1,
                write_timeout=1,
                inter_byte_timeout=0.1
            ) as ser:
                print("串口打开成功")
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                print("缓冲区已清空")
                
                ser.rts = True
                ser.dtr = True
                print("RTS/DTR已设置")
                
                self.status_label.config(text="等待接收文件...")
                print("开始等待数据...")
                
                while self.is_receiving:
                    try:
                        print(".", end="", flush=True)  # 显示心跳
                        self.receive_file(ser)
                        # 每次接收完成后添加短暂延时
                        time.sleep(0.5)
                    except Exception as e:
                        if self.is_receiving:  # 只在非主动停止时显示错误
                            print(f"\n接收错误: {str(e)}")
                            self.status_label.config(text=f"等待新的文件...")
                            # 错误后添加较长延时
                            time.sleep(1)
                        continue
                    
        except Exception as e:
            print(f"\n串口错误: {str(e)}")
            self.status_label.config(text=f"串口错误: {str(e)}")
        finally:
            self.is_receiving = False
            self.control_button.config(text="启动接收")
            self.com_box.config(state='normal')
            self.baud_box.config(state='normal')
            self.stop_current_button.config(state=tk.DISABLED)
            
    def receive_file(self, ser):
        try:
            print("\n等待接收文件...")
            
            # 接收文件名
            filename = self._receive_data(ser).decode('utf-8')
            print(f"接收到文件名: {filename}")
            self.status_label.config(text=f"接收文件: {filename}")
            # 启用中断按钮
            self.stop_current_button.config(state=tk.NORMAL)
            
            # 发送确认
            print("发送文件名确认...")
            self._send_data(ser, b"OK")
            
            # 接收文件大小
            file_size_data = self._receive_data(ser).decode('utf-8')
            print(f"接收到文件大小数据: {file_size_data}")
            file_size = int(file_size_data)
            print(f"文件大小: {file_size} bytes")
            # 发送确认
            print("发送文件大小确认...")
            self._send_data(ser, b"OK")
            
            # 生成保存路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"{timestamp}_{filename}")
            print(f"保存路径: {save_path}")
            
            # 接收文件内容
            with open(save_path, 'wb') as f:
                received_size = 0
                while received_size < file_size:
                    # 检查中断标志
                    if self.stop_current_transfer:
                        raise Exception("用户中断传输")
                        
                    print(f"\n准备接收数据块 (已接收: {received_size}/{file_size})")
                    try:
                        data = self._receive_data(ser)
                        if not data:
                            raise Exception("连接断开")
                        
                        print(f"接收到数据块: {len(data)} bytes")
                        f.write(data)
                        received_size += len(data)
                        
                        # 更新进度
                        progress = (received_size / file_size) * 100
                        self.progress['value'] = progress
                        self.status_label.config(text=f"进度: {received_size}/{file_size} bytes")
                        self.root.update()
                        
                        # 发送确认
                        print("发送数据块确认...")
                        self._send_data(ser, b"OK")
                        
                    except Exception as e:
                        print(f"接收数据块错误: {str(e)}")
                        # 尝试发送错误信息
                        try:
                            self._send_data(ser, f"ERROR: {str(e)}".encode())
                        except:
                            pass
                        raise
            
            print("文件接收完成")
            self.status_label.config(text=f"文件保存成功: {save_path}")
            
        except Exception as e:
            print(f"接收错误: {str(e)}")
            self.status_label.config(text=f"错误: {str(e)}")
        finally:
            # 重置状态
            self.stop_current_transfer = False
            self.stop_current_button.config(state=tk.DISABLED)
            self.progress['value'] = 0
            
    def _receive_data(self, ser):
        try:
            # 读取长度数据，确保读取完整的8字节
            length_data = b''
            timeout_start = time.time()
            
            while len(length_data) < 8:
                if time.time() - timeout_start > 5:  # 5秒超时
                    raise Exception("读取长度数据超时")
                    
                data = ser.read(8 - len(length_data))
                if data:
                    length_data += data
                    print(f"读取长度数据进度: {len(length_data)}/8 bytes")
            
            print(f"接收到长度数据: {length_data}")
            length = int(length_data.decode())
            print(f"准备接收数据: {length} bytes")
            
            # 读取数据，确保读取完整的数据
            data = b''
            timeout_start = time.time()
            
            while len(data) < length:
                if time.time() - timeout_start > 10:  # 10秒超时
                    raise Exception("读取数据超时")
                    
                chunk = ser.read(length - len(data))
                if chunk:
                    data += chunk
                    print(f"读取数据进度: {len(data)}/{length} bytes")
                
            print(f"接收到数据: {len(data)} bytes")
            return self.crypto.decrypt(data)
            
        except Exception as e:
            print(f"接收数据错误: {str(e)}")
            raise
    
    def _send_data(self, ser, data):
        try:
            encrypted_data = self.crypto.encrypt(data)
            length_str = f"{len(encrypted_data):08d}".encode()
            print(f"发送数据长度: {length_str}")
            ser.write(length_str)
            ser.flush()  # 确保数据发送
            time.sleep(0.1)  # 短暂延时
            
            print(f"发送加密数据: {len(encrypted_data)} bytes")
            ser.write(encrypted_data)
            ser.flush()  # 确保数据发送
            
        except Exception as e:
            print(f"发送数据错误: {str(e)}")
            raise
        
    def run(self):
        """启动接收器"""
        print("准备启动接收器...")
        
        def delayed_start():
            print("自动启动接收...")
            if not self.com_var.get():
                print("未选择串口，等待手动选择...")
                return
            self.toggle_receiving()
        
        # 启动时自动开始接收（延迟1秒）
        self.root.after(1000, delayed_start)
        print("启动主界面...")
        self.root.mainloop()

    def stop_current(self):
        """中断当前传输"""
        self.stop_current_transfer = True
        self.stop_current_button.config(state=tk.DISABLED)
        self.status_label.config(text="正在中断传输...")
        
def main():
    print("初始化接收器...")
    receiver = FileReceiver(use_encryption=False)
    print("启动接收器界面...")
    receiver.run()

if __name__ == "__main__":
    main() 