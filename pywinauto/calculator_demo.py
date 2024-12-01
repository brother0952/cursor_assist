from pywinauto.application import Application
import time

def calculate_simple_sum():
    # 启动计算器
    app = Application(backend='uia').start('calc.exe')
    
    # 等待计算器窗口完全加载
    time.sleep(1)
    
    # 获取计算器主窗口
    calculator = app.window(class_name="ApplicationFrameWindow")
    
    # 点击按钮序列：1 + 1 =
    calculator.type_keys('1')
    calculator.type_keys('+')
    calculator.type_keys('1')
    calculator.type_keys('=')
    
    # 等待一会儿以便查看结果
    time.sleep(2)
    
    # 关闭计算器
    calculator.type_keys('%{F4}')

if __name__ == "__main__":
    calculate_simple_sum() 