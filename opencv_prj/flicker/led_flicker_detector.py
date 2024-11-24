import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from opencv_prj.utils.plot_utils import setup_chinese_font, create_figure_with_chinese
from opencv_prj.utils.cv_utils import put_chinese_text

class LEDFlickerDetector:
    def __init__(self):
        self.brightness_history = deque(maxlen=100)  # 存储最近100帧的亮度值
        self.threshold = 0.1  # 频闪判定阈值
        self.cap = None
        self.show_plot = False  # 控制图表显示的开关
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        
    def setup_camera(self):
        """设置摄像头，建议使用高帧率摄像头"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 120)
        return self.cap.isOpened()
        
    def setup_plot(self):
        """设置matplotlib图表"""
        if self.fig is None:
            setup_chinese_font()
            plt.ion()  # 开启交互模式
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
    def close_plot(self):
        """关闭matplotlib图表"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax1 = None
            self.ax2 = None
    
    def toggle_plot(self):
        """切换图表显示状态"""
        self.show_plot = not self.show_plot
        if self.show_plot:
            self.setup_plot()
        else:
            self.close_plot()
            
    def update_plot(self, cv_value, fft_result):
        """更新图表显示"""
        if not self.show_plot or self.fig is None:
            return
            
        self.ax1.clear()
        self.ax1.plot(list(self.brightness_history))
        self.ax1.set_title(f'亮度变化 (变异系数: {cv_value:.3f})')
        self.ax1.set_xlabel('帧数')
        self.ax1.set_ylabel('亮度值')
        
        self.ax2.clear()
        self.ax2.plot(np.abs(fft_result[:len(fft_result)//2]))
        self.ax2.set_title('频率分析')
        self.ax2.set_xlabel('频率')
        self.ax2.set_ylabel('幅值')
        
        plt.tight_layout()
        plt.pause(0.01)
        
    def get_roi_brightness(self, frame):
        """获取ROI区域的平均亮度"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        if np.sum(thresh) == 0:
            return None
            
        mean_brightness = np.mean(gray[thresh > 0])
        return mean_brightness
        
    def analyze_flicker(self):
        """分析频闪程度"""
        if len(self.brightness_history) < 50:
            return None, 0
            
        brightness_array = np.array(self.brightness_history)
        fft_result = np.fft.fft(brightness_array)
        cv = np.std(brightness_array) / np.mean(brightness_array)
        
        return cv, np.abs(fft_result)
        
    def run(self):
        if not self.setup_camera():
            print("无法打开摄像头")
            return
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            brightness = self.get_roi_brightness(frame)
            if brightness is not None:
                self.brightness_history.append(brightness)
                
                if len(self.brightness_history) >= 50:
                    cv, fft_result = self.analyze_flicker()
                    
                    # 更新图表（如果启用）
                    self.update_plot(cv, fft_result)
                    
                    # 显示频闪状态
                    if cv > self.threshold:
                        frame = put_chinese_text(frame, "严重频闪!", (10, 30), 
                                              color=(0, 0, 255))
                    else:
                        frame = put_chinese_text(frame, "正常", (10, 30), 
                                              color=(0, 255, 0))
            
            # 显示图表开关状态
            status_text = "图表显示: 开启" if self.show_plot else "图表显示: 关闭"
            frame = put_chinese_text(frame, status_text, (10, 60), 
                                  color=(255, 255, 0))
            
            cv2.imshow('LED频闪检测', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):  # 按'p'键切换图表显示
                self.toggle_plot()
                
        self.cap.release()
        cv2.destroyAllWindows()
        self.close_plot()

if __name__ == "__main__":
    detector = LEDFlickerDetector()
    detector.run()