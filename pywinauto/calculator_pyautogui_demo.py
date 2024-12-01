import pyautogui
import time
import os

def calculate_simple_sum():
    # 启动计算器
    os.system('calc.exe')
    
    # 等待计算器启动
    time.sleep(1)
    
    # 输入计算序列：1 + 1 =
    pyautogui.press('1')
    pyautogui.press('+')
    pyautogui.press('1')
    pyautogui.press('enter')  # Windows计算器中enter键等同于等号
    
    # 等待查看结果
    time.sleep(2)
    
    # 关闭计算器 (Alt+F4)
    pyautogui.hotkey('alt', 'f4')

if __name__ == "__main__":
    # 添加一个安全措施，将鼠标移到屏幕左上角可以中断程序
    pyautogui.FAILSAFE = True
    calculate_simple_sum() 