import win32com.client
import time
import os

class CANoeConnection:
    def __init__(self):
        self.application = None
        self.measurement = None
    
    def connect(self):
        """连接到CANoe应用程序"""
        try:
            self.application = win32com.client.Dispatch("CANoe.Application")
            print("成功连接到CANoe")
            return True
        except Exception as e:
            print(f"连接CANoe失败: {str(e)}")
            return False
    
    def open_configuration(self, config_path):
        """打开CANoe配置文件"""
        if not os.path.exists(config_path):
            print(f"配置文件不存在: {config_path}")
            return False
        
        try:
            self.application.Open(config_path)
            print(f"成功打开配置文件: {config_path}")
            return True
        except Exception as e:
            print(f"打开配置文件失败: {str(e)}")
            return False
    
    def start_measurement(self):
        """开始测量"""
        try:
            self.measurement = self.application.Measurement
            self.measurement.Start()
            print("测量已启动")
            return True
        except Exception as e:
            print(f"启动测量失败: {str(e)}")
            return False
    
    def stop_measurement(self):
        """停止测量"""
        try:
            if self.measurement:
                self.measurement.Stop()
                print("测量已停止")
                return True
        except Exception as e:
            print(f"停止测量失败: {str(e)}")
            return False
    
    def get_system_variable(self, namespace, variable):
        """获取系统变量值"""
        try:
            var = self.application.System.Namespaces.Item(namespace).Variables.Item(variable)
            return var.Value
        except Exception as e:
            print(f"获取系统变量失败: {str(e)}")
            return None
    
    def set_system_variable(self, namespace, variable, value):
        """设置系统变量值"""
        try:
            var = self.application.System.Namespaces.Item(namespace).Variables.Item(variable)
            var.Value = value
            return True
        except Exception as e:
            print(f"设置系统变量失败: {str(e)}")
            return False

def main():
    # 创建示例
    canoe = CANoeConnection()
    
    # 连接到CANoe
    if not canoe.connect():
        return
    
    # 打开配置文件（请替换为实际的配置文件路径）
    config_path = r"D:\CANoe_Configs\Example.cfg"
    if not canoe.open_configuration(config_path):
        return
    
    # 启动测量
    if not canoe.start_measurement():
        return
    
    try:
        # 示例：读取和设置系统变量
        print("读取系统变量...")
        value = canoe.get_system_variable("SystemVariables", "ExampleVar")
        print(f"变量值: {value}")
        
        print("设置系统变量...")
        canoe.set_system_variable("SystemVariables", "ExampleVar", 123)
        
        # 等待一段时间
        time.sleep(5)
        
    finally:
        # 停止测量
        canoe.stop_measurement()

if __name__ == "__main__":
    main() 