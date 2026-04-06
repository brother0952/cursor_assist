import cv2
import numpy as np
import time
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
import os
try:
    import winsound
except ImportError:
    winsound = None

class TemplateLEDDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.red_template = None
        self.blue_template = None
        self.charging_start_time = None
        self.was_charging = False
        self.alert_shown = False
        self.pending_alert = None
        self.current_frame = None
        
    def load_templates(self, red_template_path='red_led_template.jpg', blue_template_path='blue_led_template.jpg'):
        """
        加载LED模板图片
        """
        if os.path.exists(red_template_path):
            self.red_template = cv2.imread(red_template_path)
            if self.red_template is not None:
                print(f"红色LED模板加载成功: {red_template_path}")
            else:
                print(f"红色LED模板加载失败: {red_template_path}")
        else:
            print(f"红色LED模板文件不存在: {red_template_path}")
            
        if os.path.exists(blue_template_path):
            self.blue_template = cv2.imread(blue_template_path)
            if self.blue_template is not None:
                print(f"蓝色LED模板加载成功: {blue_template_path}")
            else:
                print(f"蓝色LED模板加载失败: {blue_template_path}")
        else:
            print(f"蓝色LED模板文件不存在: {blue_template_path}")
            
        if self.red_template is None and self.blue_template is None:
            print("警告: 没有加载任何模板，将无法进行模板匹配检测")
    
    def put_chinese_text(self, img, text, position, font_scale=1, color=(0, 255, 0), thickness=2):
        """
        在图像上绘制中文文本
        """
        # 使用OpenCV的 Hershey 字体，虽然不支持中文，但可以正常显示英文
        # 如果需要显示中文，需要使用其他库如 PIL
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return img
    
    def match_template(self, frame, template, threshold=0.8):
        """
        在图像中匹配模板
        返回匹配位置列表
        """
        if template is None:
            return []
            
        # 转换为灰度图
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # 执行模板匹配
        result = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)
        
        # 查找匹配位置
        locations = np.where(result >= threshold)
        
        # 提取匹配点坐标
        matches = []
        template_h, template_w = gray_template.shape
        
        # 定义尺寸容差（例如，允许±10像素的差异）
        size_tolerance = 10
        
        # 增加调试信息
        print(f"模板尺寸: {template_w}x{template_h}")
        print(f"匹配阈值: {threshold}")
        print(f"匹配位置数量: {len(locations[0])}")
        
        for pt in zip(*locations[::-1]):
            # 计算中心点坐标
            center_x = pt[0] + template_w // 2
            center_y = pt[1] + template_h // 2
            
            # 获取该位置的相似度值
            similarity = result[pt[1], pt[0]]
            
            # 检查匹配区域的尺寸是否接近模板尺寸
            # 这里假设匹配区域的尺寸是固定的，但可以进一步优化
            if (template_w - size_tolerance <= template_w <= template_w + size_tolerance and 
                template_h - size_tolerance <= template_h <= template_h + size_tolerance):
                matches.append((center_x, center_y, pt[0], pt[1]))
                
                # 增加调试信息
                print(f"匹配点: ({center_x}, {center_y}), 相似度: {similarity:.3f}, 位置: ({pt[0]}, {pt[1]})")
            else:
                # 增加调试信息
                print(f"尺寸不匹配的点: ({pt[0]}, {pt[1]}), 模板尺寸: {template_w}x{template_h}")
            
        return matches
    
    def detect_leds(self, frame):
        """
        使用模板匹配检测LED
        """
        red_matches = self.match_template(frame, self.red_template)
        blue_matches = self.match_template(frame, self.blue_template)
        
        return red_matches, blue_matches
    
    def show_alert(self, duration):
        """
        存储警报信息以便在主线程中显示
        """
        self.pending_alert = duration
    
    def display_alert(self):
        """
        在主线程中显示警报
        """
        if self.pending_alert:
            # 播放声音
            if winsound:
                try:
                    winsound.MessageBeep(winsound.MB_ICONASTERISK)
                except:
                    pass  # 如果声音有问题就忽略
            
            # 在控制台打印信息
            print(f"提醒: 电池充电完成！总充电时间: {self.pending_alert}")
            self.pending_alert = None
    
    def draw_detections(self, frame, red_matches, blue_matches):
        """
        在图像上绘制检测结果
        """
        debug_frame = frame.copy()
        
        # 绘制红色LED匹配结果
        if self.red_template is not None and len(red_matches) > 0:
            template_h, template_w = self.red_template.shape[:2]
            for match in red_matches:
                center_x, center_y, x, y = match
                # 绘制矩形框
                cv2.rectangle(debug_frame, (x, y), (x + template_w, y + template_h), (0, 0, 255), 2)
                # 绘制中心点
                cv2.circle(debug_frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # 绘制蓝色LED匹配结果
        if self.blue_template is not None and len(blue_matches) > 0:
            template_h, template_w = self.blue_template.shape[:2]
            for match in blue_matches:
                center_x, center_y, x, y = match
                # 绘制矩形框
                cv2.rectangle(debug_frame, (x, y), (x + template_w, y + template_h), (255, 0, 0), 2)
                # 绘制中心点
                cv2.circle(debug_frame, (center_x, center_y), 5, (255, 0, 0), -1)
                
        return debug_frame
    
    def run(self):
        """
        主循环
        """
        # 尝试加载模板
        self.load_templates()
        
        print("使用模板匹配检测LED")
        print("按 'q' 退出")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            self.current_frame = frame.copy()
            
            # 检测LED
            red_matches, blue_matches = self.detect_leds(frame)
            # print()
            # 绘制检测结果
            debug_frame = self.draw_detections(frame, red_matches, blue_matches)
            
            # 根据LED状态判断充电状态
            status = ""
            if len(red_matches) == 1 and len(blue_matches) == 1:
                # 一个红色和一个蓝色LED = 充电中
                status = "Charging"  # 充电中
                if not self.was_charging:
                    self.charging_start_time = datetime.now()
                    self.was_charging = True
                    self.alert_shown = False
            elif len(blue_matches) == 2 and len(red_matches) == 0:
                # 两个蓝色LED = 充电完成
                status = "Charging Complete"  # 充电完成
                
                # 充电完成时显示提醒
                if self.was_charging and not self.alert_shown:
                    if self.charging_start_time:
                        duration = datetime.now() - self.charging_start_time
                        minutes, seconds = divmod(duration.seconds, 60)
                        hours, minutes = divmod(minutes, 60)
                        
                        if hours > 0:
                            duration_str = f"{hours}h {minutes}m {seconds}s"
                        else:
                            duration_str = f"{minutes}m {seconds}s"
                        
                        # 显示提醒
                        self.show_alert(duration_str)
                        self.alert_shown = True
                self.was_charging = False
            else:
                # 处理中间状态
                if len(blue_matches) >= 1 and len(red_matches) == 0:
                    status = "Waiting for charging"  # 等待充电
                else:
                    status = f"b:{len(blue_matches)} ,r:{len(red_matches)}Waiting for device..."  # 等待设备...
                # 只有在没有LED时才重置充电状态
                if len(blue_matches) == 0 and len(red_matches) == 0:
                    self.was_charging = False
            
            # 显示提醒
            self.display_alert()
            
            # 在画面上显示状态
            cv2.putText(debug_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示说明
            cv2.putText(debug_frame, "Press 'q' to quit", (10, debug_frame.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Template LED Detector', debug_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = TemplateLEDDetector()
    detector.run()