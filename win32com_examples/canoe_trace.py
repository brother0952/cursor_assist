import win32com.client
import time
from datetime import datetime

class CANoeTrace:
    def __init__(self):
        self.application = None
        self.measurement = None
        self.trace_window = None
    
    def connect(self):
        """连接到CANoe并初始化跟踪窗口"""
        try:
            self.application = win32com.client.Dispatch("CANoe.Application")
            self.measurement = self.application.Measurement
            print("成功连接到CANoe")
            return True
        except Exception as e:
            print(f"连接CANoe失败: {str(e)}")
            return False
    
    def initialize_trace(self):
        """初始化跟踪窗口"""
        try:
            self.trace_window = self.application.UI.Windows.Item("Trace")
            if not self.trace_window.Visible:
                self.trace_window.Visible = True
            print("跟踪窗口已初始化")
            return True
        except Exception as e:
            print(f"初始化跟踪窗口失败: {str(e)}")
            return False
    
    def configure_trace_filter(self, channel=None, msg_id=None, msg_name=None):
        """配置跟踪过滤器"""
        try:
            filter_config = self.trace_window.FilterConfiguration
            
            # 清除现有过滤器
            filter_config.Clear()
            
            if channel is not None:
                filter_config.SetChannel(channel)
            if msg_id is not None:
                filter_config.SetId(msg_id)
            if msg_name is not None:
                filter_config.SetMessageName(msg_name)
            
            filter_config.Apply()
            print("跟踪过滤器已配置")
            return True
        except Exception as e:
            print(f"配置跟踪过滤器失败: {str(e)}")
            return False
    
    def start_trace_recording(self, file_path):
        """开始记录跟踪数据"""
        try:
            self.trace_window.WriteToFile(file_path, True)  # True表示追加模式
            print(f"开始记录跟踪数据到: {file_path}")
            return True
        except Exception as e:
            print(f"开始记录跟踪数据失败: {str(e)}")
            return False
    
    def stop_trace_recording(self):
        """停止记录跟踪数据"""
        try:
            self.trace_window.StopWriteToFile()
            print("停止记录跟踪数据")
            return True
        except Exception as e:
            print(f"停止记录跟踪数据失败: {str(e)}")
            return False
    
    def clear_trace(self):
        """清除跟踪窗口内容"""
        try:
            self.trace_window.Clear()
            print("跟踪窗口已清除")
            return True
        except Exception as e:
            print(f"清除跟踪窗口失败: {str(e)}")
            return False
    
    def get_trace_entries(self, count=10):
        """获取最近的跟踪条目"""
        try:
            entries = []
            total_entries = self.trace_window.EntryCount
            start_index = max(0, total_entries - count)
            
            for i in range(start_index, total_entries):
                entry = self.trace_window.GetEntry(i)
                entries.append({
                    'time': entry.Time,
                    'channel': entry.Channel,
                    'msg_name': entry.MessageName,
                    'msg_id': entry.ID,
                    'data': entry.DataString
                })
            
            return entries
        except Exception as e:
            print(f"获取跟踪条目失败: {str(e)}")
            return None

def main():
    # 创建示例
    canoe = CANoeTrace()
    
    # 连接到CANoe
    if not canoe.connect():
        return
    
    try:
        # 初始化跟踪窗口
        if not canoe.initialize_trace():
            return
        
        # 示例1：配置过滤器
        print("\n示例1：配置跟踪过滤器")
        canoe.configure_trace_filter(channel=1, msg_name="EngineData")
        
        # 示例2：开始记录
        print("\n示例2：开始记录跟踪数据")
        canoe.start_trace_recording("trace_log.txt")
        
        # 等待一段时间收集数据
        print("收集数据中...")
        time.sleep(10)
        
        # 示例3：获取跟踪条目
        print("\n示例3：获取最近的跟踪条目")
        entries = canoe.get_trace_entries(5)
        if entries:
            for entry in entries:
                print(f"时间: {entry['time']}")
                print(f"通道: {entry['channel']}")
                print(f"消息: {entry['msg_name']} (ID: {entry['msg_id']})")
                print(f"数据: {entry['data']}")
                print("---")
        
        # 停止记录
        canoe.stop_trace_recording()
        
        # 清除跟踪窗口
        canoe.clear_trace()
        
    except Exception as e:
        print(f"执行过程中出错: {str(e)}")

if __name__ == "__main__":
    main() 