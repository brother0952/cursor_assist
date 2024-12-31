import serial
import serial.tools.list_ports
import os
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from crypto import Crypto
import threading
import time
from io import BytesIO
from config_manager import ConfigManager

class FileReceiver:
    def __init__(self, use_encryption=False):
        self.root = tk.Tk()
        self.root.title("文件接收器")
        self.root.geometry("400x300")
        
        # 加载配置
        self.config = ConfigManager("receiver_config.json")
        
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
        self.com_var = tk.StringVar(value=self.config.get("port", ""))
        self.com_box = ttk.Combobox(port_row, textvariable=self.com_var, width=15)
        self.refresh_ports()
        self.com_box.pack(side=tk.LEFT, padx=2)
        ttk.Button(port_row, text="刷新", command=self.refresh_ports, width=8).pack(side=tk.LEFT, padx=2)
        
        # 波特率选择
        baud_row = ttk.Frame(port_frame)
        baud_row.pack(fill=tk.X, pady=2)
        ttk.Label(baud_row, text="波特率:").pack(side=tk.LEFT, padx=2)
        self.baud_var = tk.StringVar(value=self.config.get("baudrate", "115200"))
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
        
        # 添加配置变更监听
        self.com_var.trace('w', self.on_port_change)
        self.baud_var.trace('w', self.on_baudrate_change)

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
                # 禁用流控制以减少开销
                xonxoff=False,    # 禁用软件流控
                rtscts=False,     # 禁用硬件流控
                dsrdtr=False      # 禁用DSR/DTR
            ) as ser:
                print("串口打开成功")
                
                # 设置RTS/CTS
                ser.rts = True
                ser.dtr = True
                print("RTS/DTR已设置")
                
                # 清空缓冲区
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                print("缓冲区已清空")
                
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
            
            # 接收文件信息
            filename = self._receive_data(ser).decode('utf-8')
            file_size = int(self._receive_data(ser).decode('utf-8'))
            print(f"接收文件: {filename} ({file_size} bytes)")
            
            # 生成保存路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"{timestamp}_{filename}")
            print(f"保存路径: {save_path}")
            
            # 使用临时文件
            temp_path = save_path + ".tmp"
            received_size = 0
            
            with open(temp_path, 'wb') as f:
                while received_size < file_size:
                    try:
                        # 接收数据块
                        data = self._receive_data(ser)
                        if not data:
                            raise Exception("接收到空数据")
                        
                        # 写入文件
                        f.write(data)
                        received_size += len(data)
                        
                        # 发送确认
                        self._send_data(ser, b"OK")
                        
                        # 更新进度
                        progress = (received_size / file_size) * 100
                        self.progress['value'] = progress
                        self.status_label.config(text=f"进度: {received_size}/{file_size} bytes")
                        self.root.update()
                        
                    except Exception as e:
                        print(f"接收数据块错误: {str(e)}")
                        try:
                            self._send_data(ser, f"ERROR: {str(e)}".encode())
                        except:
                            pass
                        raise
            
            # 接收完成后重命名文件
            os.rename(temp_path, save_path)
            print("文件接收完成")
            self.status_label.config(text=f"文件保存成功: {save_path}")
            
        except Exception as e:
            print(f"接收错误: {str(e)}")
            self.status_label.config(text=f"错误: {str(e)}")
            # 删除临时文件
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    def _receive_data(self, ser):
        try:
            # 读取长度数据
            length_data = ser.read(8)  # 直接读取8字节
            if len(length_data) != 8:
                raise Exception("读取长度数据超时")
            
            length = int(length_data.decode())
            print(f"准备接收数据: {length} bytes")
            
            # 一次性读取所有数据
            data = ser.read(length)
            if len(data) != length:
                raise Exception("读取数据不完整")
            
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
        
    def on_port_change(self, *args):
        """串口变更时保存配置"""
        self.config.set("port", self.com_var.get())
        
    def on_baudrate_change(self, *args):
        """波特率变更时保存配置"""
        self.config.set("baudrate", self.baud_var.get())

def main():
    print("初始化接收器...")
    receiver = FileReceiver(use_encryption=False)
    print("启动接收器界面...")
    receiver.run()

if __name__ == "__main__":
    main() 