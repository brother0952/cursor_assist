import pyautogui
import time
from typing import Optional

class DoorsAutomation:
    def __init__(self):
        # 设置pyautogui的安全属性
        pyautogui.FAILSAFE = True
        # 设置操作间隔，DOORS响应可能较慢
        pyautogui.PAUSE = 0.5
    
    def wait_and_click(self, image_path: str, timeout: int = 10) -> Optional[tuple]:
        """
        等待并点击指定图像
        :param image_path: 要查找的图像路径
        :param timeout: 超时时间（秒）
        :return: 如果找到并点击则返回坐标，否则返回None
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                location = pyautogui.locateCenterOnScreen(image_path, confidence=0.9)
                if location:
                    pyautogui.click(location)
                    return location
            except pyautogui.ImageNotFoundException:
                time.sleep(0.5)
        return None

    def insert_object(self):
        try:
            # 确保DOORS窗口处于活动状态
            pyautogui.hotkey('alt', 'tab')
            time.sleep(1)
            
            # 点击Edit菜单
            pyautogui.hotkey('alt', 'e')
            time.sleep(0.5)
            
            # 选择Create Object选项
            # 方法1：使用快捷键
            pyautogui.hotkey('alt', 'o')
            
            # 方法2：或者使用鼠标移动（需要提前截图菜单项）
            # self.wait_and_click('images/create_object.png')
            
            # 等待对象创建对话框出现
            time.sleep(1)
            
            # 输入对象内容
            pyautogui.write('New Object Content')
            
            # 按Enter确认创建
            pyautogui.press('enter')
            
            print("成功创建新对象")
            
        except Exception as e:
            print(f"操作失败: {str(e)}")

def main():
    doors = DoorsAutomation()
    
    # 给用户一些时间切换到DOORS窗口
    print("请在5秒内切换到DOORS窗口...")
    time.sleep(5)
    
    # 执行插入对象操作
    doors.insert_object()

if __name__ == "__main__":
    main() 