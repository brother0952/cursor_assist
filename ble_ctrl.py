import ctypes
import subprocess
import time

class BluetoothController:
    def __init__(self):
        self.BLUETOOTH_RADIO_STATE = {
            0: "Off",
            1: "On",
            2: "Disabled",
            3: "Unknown"
        }

    def get_bluetooth_state(self):
        """获取蓝牙状态"""
        try:
            # 使用 PowerShell 获取蓝牙适配器状态
            cmd = "Get-PnpDevice | Where-Object {$_.Class -eq 'Bluetooth'} | Select-Object Status"
            result = subprocess.check_output(['powershell', '-Command', cmd], text=True)
            return "On" if "OK" in result else "Off"
        except Exception as e:
            print(f"获取蓝牙状态失败: {e}")
            return "Unknown"

    def enable_bluetooth(self):
        """启用蓝牙"""
        try:
            # 启用蓝牙服务
            subprocess.run(["powershell", "-Command", "Start-Service bthserv"], check=True)
            time.sleep(1)  # 等待服务启动
            
            # 启用蓝牙适配器
            cmd = """
            $bluetooth = Get-PnpDevice | Where-Object {$_.Class -eq 'Bluetooth'};
            Enable-PnpDevice -InstanceId $bluetooth.InstanceId -Confirm:$false
            """
            subprocess.run(["powershell", "-Command", cmd], check=True)
            print("蓝牙已启用")
            return True
        except Exception as e:
            print(f"启用蓝牙失败: {e}")
            return False

    def disable_bluetooth(self):
        """禁用蓝牙"""
        try:
            # 禁用蓝牙适配器
            cmd = """
            $bluetooth = Get-PnpDevice | Where-Object {$_.Class -eq 'Bluetooth'};
            Disable-PnpDevice -InstanceId $bluetooth.InstanceId -Confirm:$false
            """
            subprocess.run(["powershell", "-Command", cmd], check=True)
            
            # 停止蓝牙服务
            subprocess.run(["powershell", "-Command", "Stop-Service bthserv"], check=True)
            print("蓝牙已禁用")
            return True
        except Exception as e:
            print(f"禁用蓝牙失败: {e}")
            return False

    def toggle_bluetooth(self):
        """切换蓝牙状态"""
        current_state = self.get_bluetooth_state()
        if current_state == "Off":
            return self.enable_bluetooth()
        elif current_state == "On":
            return self.disable_bluetooth()
        else:
            print(f"无法切换蓝牙状态，当前状态: {current_state}")
            return False

def main():
    controller = BluetoothController()
    
    # 显示当前状态
    print(f"当前蓝牙状态: {controller.get_bluetooth_state()}")
    
    # 提供命令行交互
    while True:
        command = input("\n请输入命令 (on/off/toggle/status/exit): ").lower()
        
        if command == "on":
            controller.enable_bluetooth()
        elif command == "off":
            controller.disable_bluetooth()
        elif command == "toggle":
            controller.toggle_bluetooth()
        elif command == "status":
            print(f"蓝牙状态: {controller.get_bluetooth_state()}")
        elif command == "exit":
            break
        else:
            print("无效命令。可用命令: on, off, toggle, status, exit")


'''
注意事项：
需要管理员权限
可能需要在 Windows 设置中允许脚本控制蓝牙
某些操作可能需要重启蓝牙服务
如果遇到权限问题，可以：
以管理员身份运行 PowerShell
执行 Set-ExecutionPolicy RemoteSigned 允许运行本地脚本
然后再运行 Python 脚本
'''
if __name__ == "__main__":
    main()