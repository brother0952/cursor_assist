import win32com.client
import time
from datetime import datetime

class CANoeSignals:
    def __init__(self):
        self.application = None
        self.measurement = None
        self.env = None
    
    def connect(self):
        """连接到CANoe并初始化环境"""
        try:
            self.application = win32com.client.Dispatch("CANoe.Application")
            self.measurement = self.application.Measurement
            self.env = self.application.Environment
            print("成功连接到CANoe")
            return True
        except Exception as e:
            print(f"连接CANoe失败: {str(e)}")
            return False
    
    def get_signal_value(self, channel, msg_name, signal_name):
        """获取指定信号的值"""
        try:
            signal = self.env.GetSignal(channel, msg_name, signal_name)
            return {
                'value': signal.Value,
                'timestamp': signal.TimeStamp,
                'raw_value': signal.RawValue
            }
        except Exception as e:
            print(f"获取信号值失败: {str(e)}")
            return None
    
    def set_signal_value(self, channel, msg_name, signal_name, value):
        """设置指定信号的值"""
        try:
            signal = self.env.GetSignal(channel, msg_name, signal_name)
            signal.Value = value
            return True
        except Exception as e:
            print(f"设置信号值失败: {str(e)}")
            return False
    
    def monitor_signal(self, channel, msg_name, signal_name, duration=10):
        """监控信号值变化"""
        print(f"开始监控信号: {msg_name}.{signal_name}")
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                signal_info = self.get_signal_value(channel, msg_name, signal_name)
                if signal_info:
                    timestamp = datetime.fromtimestamp(signal_info['timestamp'])
                    print(f"时间: {timestamp}, 值: {signal_info['value']}, 原始值: {signal_info['raw_value']}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("监控被用户中断")
    
    def send_cyclic_signal(self, channel, msg_name, signal_name, values, interval=1.0):
        """循环发送信号值"""
        print(f"开始循环发送信号: {msg_name}.{signal_name}")
        try:
            while True:
                for value in values:
                    if self.set_signal_value(channel, msg_name, signal_name, value):
                        print(f"发送值: {value}")
                    time.sleep(interval)
        except KeyboardInterrupt:
            print("循环发送被用户中断")

def main():
    # 创建示例
    canoe = CANoeSignals()
    
    # 连接到CANoe
    if not canoe.connect():
        return
    
    try:
        # 示例1：读取信号值
        print("\n示例1：读取信号值")
        signal_info = canoe.get_signal_value(1, "EngineData", "EngineSpeed")
        if signal_info:
            print(f"发动机转速: {signal_info['value']} RPM")
        
        # 示例2：设置信号值
        print("\n示例2：设置信号值")
        canoe.set_signal_value(1, "EngineData", "EngineSpeed", 2500)
        
        # 示例3：监控信号（10秒）
        print("\n示例3：监控信号")
        canoe.monitor_signal(1, "EngineData", "EngineSpeed", 10)
        
        # 示例4：循环发送信号
        print("\n示例4：循环发送信号")
        test_values = [1000, 2000, 3000, 4000, 3000, 2000]
        canoe.send_cyclic_signal(1, "EngineData", "EngineSpeed", test_values, 0.5)
        
    except Exception as e:
        print(f"执行过程中出错: {str(e)}")

if __name__ == "__main__":
    main() 