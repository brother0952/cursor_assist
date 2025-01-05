import win32com.client
import time
from datetime import datetime

class CANoeBusStatistics:
    def __init__(self):
        self.application = None
        self.measurement = None
        self.bus_statistics = None
    
    def connect(self):
        """连接到CANoe并初始化总线统计"""
        try:
            self.application = win32com.client.Dispatch("CANoe.Application")
            self.measurement = self.application.Measurement
            print("成功连接到CANoe")
            return True
        except Exception as e:
            print(f"连接CANoe失败: {str(e)}")
            return False
    
    def initialize_statistics(self, channel):
        """初始化指定通道的总线统计"""
        try:
            self.bus_statistics = self.application.Bus.GetStatistics(channel)
            print(f"已初始化通道 {channel} 的总线统计")
            return True
        except Exception as e:
            print(f"初始化总线统计失败: {str(e)}")
            return False
    
    def get_bus_load(self):
        """获取总线负载"""
        try:
            if self.bus_statistics:
                return {
                    'current': self.bus_statistics.BusLoad,
                    'peak': self.bus_statistics.PeakLoad,
                    'average': self.bus_statistics.AverageLoad
                }
            return None
        except Exception as e:
            print(f"获取总线负载失败: {str(e)}")
            return None
    
    def get_error_statistics(self):
        """获取错误统计"""
        try:
            if self.bus_statistics:
                return {
                    'total_errors': self.bus_statistics.TotalErrorCount,
                    'error_frames': self.bus_statistics.ErrorFrameCount,
                    'controller_warnings': self.bus_statistics.ControllerWarnings
                }
            return None
        except Exception as e:
            print(f"获取错误统计失败: {str(e)}")
            return None
    
    def get_message_statistics(self):
        """获取消息统计"""
        try:
            if self.bus_statistics:
                return {
                    'total_frames': self.bus_statistics.TotalFrameCount,
                    'data_frames': self.bus_statistics.DataFrameCount,
                    'remote_frames': self.bus_statistics.RemoteFrameCount,
                    'std_frames': self.bus_statistics.StandardFrameCount,
                    'ext_frames': self.bus_statistics.ExtendedFrameCount
                }
            return None
        except Exception as e:
            print(f"获取消息统计失败: {str(e)}")
            return None
    
    def monitor_bus_statistics(self, duration=30, interval=1.0):
        """监控总线统计"""
        print(f"开始监控总线统计，持续 {duration} 秒...")
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                current_time = datetime.now().strftime("%H:%M:%S")
                
                # 获取总线负载
                load = self.get_bus_load()
                if load:
                    print(f"\n时间: {current_time}")
                    print(f"当前负载: {load['current']:.2f}%")
                    print(f"峰值负载: {load['peak']:.2f}%")
                    print(f"平均负载: {load['average']:.2f}%")
                
                # 获取错误统计
                errors = self.get_error_statistics()
                if errors:
                    print(f"总错误数: {errors['total_errors']}")
                    print(f"错误帧数: {errors['error_frames']}")
                    print(f"控制器警告: {errors['controller_warnings']}")
                
                # 获取消息统计
                messages = self.get_message_statistics()
                if messages:
                    print(f"总帧数: {messages['total_frames']}")
                    print(f"数据帧: {messages['data_frames']}")
                    print(f"远程帧: {messages['remote_frames']}")
                    print(f"标准帧: {messages['std_frames']}")
                    print(f"扩展帧: {messages['ext_frames']}")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n监控被用户中断")
    
    def reset_statistics(self):
        """重置统计数据"""
        try:
            if self.bus_statistics:
                self.bus_statistics.Reset()
                print("统计数据已重置")
                return True
            return False
        except Exception as e:
            print(f"重置统计数据失败: {str(e)}")
            return False

def main():
    # 创建示例
    canoe = CANoeBusStatistics()
    
    # 连接到CANoe
    if not canoe.connect():
        return
    
    try:
        # 初始化通道1的总线统计
        if not canoe.initialize_statistics(1):
            return
        
        # 重置统计数据
        canoe.reset_statistics()
        
        # 示例1：获取当前总线状态
        print("\n示例1：获取当前总线状态")
        load = canoe.get_bus_load()
        if load:
            print(f"当前总线负载: {load['current']:.2f}%")
        
        # 示例2：获取错误统计
        print("\n示例2：获取错误统计")
        errors = canoe.get_error_statistics()
        if errors:
            print(f"总错误数: {errors['total_errors']}")
        
        # 示例3：监控总线统计（30秒）
        print("\n示例3：监控总线统计")
        canoe.monitor_bus_statistics(30)
        
    except Exception as e:
        print(f"执行过程中出错: {str(e)}")

if __name__ == "__main__":
    main() 