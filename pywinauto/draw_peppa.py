import pyautogui
import time
import os
import math
from pynput import keyboard
import subprocess

class PeppaPigDrawing:
    def __init__(self):
        self.is_running = True
        pyautogui.FAILSAFE = True
        pyautogui.MINIMUM_DURATION = 0.01
        pyautogui.PAUSE = 0.01
        
        screen_width, screen_height = pyautogui.size()
        # 考虑功能区和标题栏的高度
        toolbar_height = 150  # 估计值，包括标题栏和功能区
        canvas_top = toolbar_height
        canvas_height = screen_height - toolbar_height
        canvas_center_y = canvas_top + (canvas_height // 2)
        
        self.start_x = screen_width // 3
        self.start_y = canvas_center_y
        
        print(f"画布大小: {screen_width}x{screen_height}")
        print(f"起始位置: ({self.start_x}, {self.start_y})")
        
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        try:
            if key == keyboard.Key.esc:
                print("检测到ESC键，准备退出...")
                self.is_running = False
                self.check_exit()
        except Exception as e:
            print(f"键盘监听错误: {str(e)}")

    def check_exit(self):
        """检查是否需要退出"""
        if not self.is_running:
            print("正在退出程序...")
            try:
                pyautogui.mouseUp(button='left')
            except:
                pass
            self.listener.stop()
            os._exit(0)

    def wait_with_exit_check(self, seconds):
        """带退出检查的等待函数"""
        print(f"等待 {seconds} 秒...")
        start_time = time.time()
        while time.time() - start_time < seconds:
            self.check_exit()
            time.sleep(0.05)
        print("等待完成")

    def open_paint(self):
        """打开画板"""
        print("正在启动画板...")
        pyautogui.hotkey('win', 'r')
        self.wait_with_exit_check(0.5)
        pyautogui.write('mspaint')
        pyautogui.press('enter')
        self.wait_with_exit_check(2)
        
        # 最大化窗口 - 使用多种方法确保成功
        print("最大化窗口...")
        # 方法1：使用Alt+Space, X
        pyautogui.hotkey('alt', 'space')
        self.wait_with_exit_check(0.5)
        pyautogui.press('x')
        self.wait_with_exit_check(1)
        
        # 方法2：再次尝试Win+Up
        pyautogui.hotkey('win', 'up')
        self.wait_with_exit_check(1)
        
        # 调整画布视图
        print("调整画布视图...")
        # 按住Ctrl并滚动鼠标滚轮以调整画布大小为100%
        pyautogui.hotkey('ctrl', '0')  # 重置缩放
        self.wait_with_exit_check(0.5)
        
        # 确保功能区完全展开
        pyautogui.press('alt')
        self.wait_with_exit_check(0.5)
        pyautogui.press('h')  # 选择Home标签
        self.wait_with_exit_check(0.5)
        
        # 选择铅笔工具
        print("选择铅笔工具...")
        pyautogui.press('p')
        self.wait_with_exit_check(0.5)
        
        # 调整起始位置
        print("调整起始位置...")
        # 移动到画布中心位置
        screen_width, screen_height = pyautogui.size()
        # 考虑功能区和标题栏的高度
        toolbar_height = 150  # 估计值，包括标题栏和功能区
        canvas_top = toolbar_height
        canvas_height = screen_height - toolbar_height
        canvas_center_y = canvas_top + (canvas_height // 5)
        
        self.start_x = screen_width // 6
        self.start_y = canvas_center_y
        
        print(f"新的起始位置: ({self.start_x}, {self.start_y})")
        pyautogui.moveTo(self.start_x, self.start_y)
        self.wait_with_exit_check(0.5)

    def draw_head(self):
        """画头部（完美圆形）"""
        print("开始画头...")
        pyautogui.moveTo(self.start_x, self.start_y)
        self.wait_with_exit_check(0.2)
        pyautogui.mouseDown(button='left')
        
        radius = 80  # 头部半径
        for i in range(360):
            self.check_exit()
            angle = math.radians(i)
            x = self.start_x + math.cos(angle) * radius
            y = self.start_y + math.sin(angle) * radius
            pyautogui.moveTo(x, y, duration=0.0001)
        
        pyautogui.mouseUp(button='left')
        print("头部绘制完成")

    def draw_ears(self):
        """画耳朵（倒U形）"""
        print("开始画耳朵...")
        
        # 左耳朵
        ear_left_x = self.start_x - 30
        ear_left_y = self.start_y - 80
        
        pyautogui.moveTo(ear_left_x, ear_left_y)
        self.wait_with_exit_check(0.2)
        pyautogui.mouseDown(button='left')
        # 画倒U形
        for i in range(180):
            self.check_exit()
            angle = math.radians(i + 180)  # 从180度开始画
            x = ear_left_x + math.cos(angle) * 20
            y = ear_left_y + math.sin(angle) * 30
            pyautogui.moveTo(x, y, duration=0.0001)
        pyautogui.mouseUp(button='left')
        
        # 右耳朵
        ear_right_x = self.start_x + 30
        ear_right_y = self.start_y - 80
        
        pyautogui.moveTo(ear_right_x, ear_right_y)
        self.wait_with_exit_check(0.2)
        pyautogui.mouseDown(button='left')
        # 画倒U形
        for i in range(180):
            self.check_exit()
            angle = math.radians(i + 180)
            x = ear_right_x + math.cos(angle) * 20
            y = ear_right_y + math.sin(angle) * 30
            pyautogui.moveTo(x, y, duration=0.0001)
        pyautogui.mouseUp(button='left')
        print("耳朵绘制完成")

    def draw_eyes(self):
        """画眼睛（小圆点）"""
        print("开始画眼睛...")
        # 眼睛位置靠近一起
        eye_y = self.start_y - 20
        
        # 左眼
        eye_left_x = self.start_x + 20
        pyautogui.moveTo(eye_left_x, eye_y)
        self.wait_with_exit_check(0.2)
        pyautogui.mouseDown(button='left')
        for i in range(360):
            self.check_exit()
            angle = math.radians(i)
            x = eye_left_x + math.cos(angle) * 3
            y = eye_y + math.sin(angle) * 3
            pyautogui.moveTo(x, y, duration=0.0001)
        pyautogui.mouseUp(button='left')
        
        # 右眼（靠近左眼）
        eye_right_x = self.start_x + 40
        pyautogui.moveTo(eye_right_x, eye_y)
        self.wait_with_exit_check(0.2)
        pyautogui.mouseDown(button='left')
        for i in range(360):
            self.check_exit()
            angle = math.radians(i)
            x = eye_right_x + math.cos(angle) * 3
            y = eye_y + math.sin(angle) * 3
            pyautogui.moveTo(x, y, duration=0.0001)
        pyautogui.mouseUp(button='left')
        print("眼睛绘制完成")

    def draw_nose(self):
        """画鼻子（更长的椭圆）"""
        print("开始画鼻子...")
        nose_x = self.start_x + 80  # 更靠右
        nose_y = self.start_y
        
        pyautogui.moveTo(nose_x, nose_y)
        self.wait_with_exit_check(0.2)
        pyautogui.mouseDown(button='left')
        
        # 画更长的椭圆形鼻子
        for i in range(360):
            self.check_exit()
            angle = math.radians(i)
            x = nose_x + math.cos(angle) * 40  # 更长
            y = nose_y + math.sin(angle) * 25
            pyautogui.moveTo(x, y, duration=0.0001)
        
        pyautogui.mouseUp(button='left')
        print("鼻子绘制完成")

    def draw_mouth(self):
        """画嘴巴（简单的水平线）"""
        print("开始画嘴巴...")
        mouth_x = self.start_x + 20
        mouth_y = self.start_y + 20
        
        pyautogui.moveTo(mouth_x, mouth_y)
        self.wait_with_exit_check(0.2)
        pyautogui.mouseDown(button='left')
        pyautogui.moveTo(mouth_x + 60, mouth_y, duration=0.0001)
        pyautogui.mouseUp(button='left')
        print("嘴巴绘制完成")

    def draw_peppa(self):
        """画完整的佩奇"""
        try:
            self.open_paint()
            
            print("正在开始绘画...")
            print("按ESC键可以随时退出程序")
            self.wait_with_exit_check(1)
            
            if self.is_running:
                self.draw_head()
            if self.is_running:
                self.wait_with_exit_check(0.3)
                self.draw_ears()  # 添加耳朵
            if self.is_running:
                self.wait_with_exit_check(0.3)
                self.draw_nose()
            if self.is_running:
                self.wait_with_exit_check(0.3)
                self.draw_eyes()
            if self.is_running:
                self.wait_with_exit_check(0.3)
                self.draw_mouth()
            
            print("绘画完成！")
            
        except Exception as e:
            print(f"绘画过程出错: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            if self.is_running:
                self.listener.stop()

def main():
    try:
        peppa = PeppaPigDrawing()
        print("将在3秒后开始绘画...")
        print("请不要移动鼠标和键盘...")
        print("按ESC键可以随时退出程序")
        
        peppa.wait_with_exit_check(3)
        if peppa.is_running:
            print("开始执行绘画程序...")
            peppa.draw_peppa()
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 