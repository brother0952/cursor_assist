import ctypes
from ctypes import windll, byref, Structure, c_uint, POINTER, c_bool
import time

class RadioState(Structure):
    _fields_ = [("RadioState", c_uint)]

class BluetoothRadioController:
    def __init__(self):
        # 加载 Windows Radio Management API
        self.rmd = windll.LoadLibrary("RMapi.dll")
        
    def get_radio_state(self):
        """获取蓝牙无线电状态"""
        state = RadioState()
        result = self.rmd.GetRadioState(2, byref(state))  # 2 代表蓝牙
        if result == 0:  # S_OK
            return "On" if state.RadioState == 1 else "Off"
        return "Unknown"
    
    def set_radio_state(self, enable: bool):
        """设置蓝牙无线电状态"""
        try:
            result = self.rmd.SetRadioState(2, c_bool(enable))
            if result == 0:  # S_OK
                print(f"蓝牙已{'启用' if enable else '禁用'}")
                return True
            else:
                print(f"设置蓝牙状态失败，错误代码: {result}")
                return False
        except Exception as e:
            print(f"设置蓝牙状态��出错: {e}")
            return False

def main():
    controller = BluetoothRadioController()
    
    while True:
        command = input("\n请输入命令 (on/off/status/exit): ").lower()
        
        if command == "on":
            controller.set_radio_state(True)
        elif command == "off":
            controller.set_radio_state(False)
        elif command == "status":
            print(f"蓝牙状态: {controller.get_radio_state()}")
        elif command == "exit":
            break
        else:
            print("无效命令。可用命令: on, off, status, exit")

if __name__ == "__main__":
    main() 