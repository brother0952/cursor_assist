import cv2
import numpy as np
from collections import deque
import threading
import time

# 尝试导入scipy用于计算频谱图
try:
    from scipy.signal import spectrogram
    HAS_SCIPY = True
except ImportError:
    print("警告: 未安装scipy，将使用简化版频谱分析")
    HAS_SCIPY = False

class VideoFFTAnalyzer:
    def __init__(self, roi_radius=30, buffer_size=200):
        """
        初始化视频FFT分析器
        
        参数:
        roi_radius: ROI区域半径（像素）
        buffer_size: 用于频谱分析的时间缓冲区大小
        """
        self.roi_radius = roi_radius
        self.buffer_size = buffer_size
        
        # 存储视频帧和时间序列数据
        self.frame = None
        self.display_frame = None
        self.gray_frame = None
        self.brightness_series = deque(maxlen=buffer_size)
        self.times = deque(maxlen=buffer_size)
        self.click_point = None
        
        # 视频捕获
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")
        
        # FPS计算
        self.last_time = time.time()
        self.fps = 30.0
        
        # 同步机制
        self.lock = threading.Lock()
        self.running = True
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数，用于设置ROI中心点"""
        if event == cv2.EVENT_LBUTTONDOWN:
            with self.lock:
                self.click_point = (x, y)
    
    def capture_frames(self):
        """在单独线程中捕获视频帧"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame.copy()
                    self.display_frame = frame.copy()
                    self.gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # 记录时间和亮度数据
                    current_time = time.time()
                    self.times.append(current_time)
                    
                    if self.click_point is not None:
                        # 绘制ROI圆
                        x, y = self.click_point
                        cv2.circle(self.display_frame, (x, y), self.roi_radius, (0, 0, 255), 2)
                        
                        # 计算ROI区域的平均亮度
                        h, w = self.gray_frame.shape
                        
                        # 确保ROI在图像范围内
                        x1 = max(0, x - self.roi_radius)
                        x2 = min(w, x + self.roi_radius)
                        y1 = max(0, y - self.roi_radius)
                        y2 = min(h, y + self.roi_radius)
                        
                        # 计算ROI区域的平均亮度
                        roi = self.gray_frame[y1:y2, x1:x2]
                        avg_brightness = np.mean(roi) if roi.size > 0 else 0
                        self.brightness_series.append(avg_brightness)
                    
                    # 更新FPS估计
                    self.fps = 1.0 / (current_time - self.last_time) if current_time > self.last_time else self.fps
                    self.last_time = current_time

    def draw_spectrogram_opencv(self, frame):
        """使用OpenCV直接在视频帧上绘制频谱图"""
        if len(self.brightness_series) < 10:
            return frame
            
        # 转换为numpy数组
        brightness_array = np.array(self.brightness_series)
        
        # 如果数据太少则返回
        if len(brightness_array) < 10:
            return frame
            
        # 移除直流分量（均值）
        brightness_array = brightness_array - np.mean(brightness_array)
        
        if HAS_SCIPY:
            # 使用scipy计算频谱图
            try:
                frequencies, times, Sxx = spectrogram(
                    brightness_array,
                    fs=self.fps,
                    nperseg=min(128, len(brightness_array)),
                    noverlap=min(64, len(brightness_array)//2),
                    scaling='spectrum',
                    window='hann',
                    mode='psd'
                
                )
                
                # 创建频谱图像
                if Sxx.shape[1] > 0:
                    # 归一化频谱数据
                    Sxx_norm = Sxx / np.max(Sxx) if np.max(Sxx) > 0 else Sxx
                    
                    # 转换为8位图像
                    spec_img = (Sxx_norm * 255).astype(np.uint8)
                    
                    # 调整图像大小以适应显示区域
                    h, w = frame.shape[:2]
                    spec_h, spec_w = spec_img.shape
                    display_width = w // 3
                    display_height = h // 3
                    
                    # 调整频谱图大小
                    resized_spec = cv2.resize(spec_img, (display_width, display_height))
                    colored_spec = cv2.cvtColor(resized_spec, cv2.COLOR_GRAY2BGR)
                    
                    # 将频谱图放在视频帧的右上角
                    frame[10:10+display_height, w-display_width-10:w-10] = colored_spec
                    
                    # 添加标签
                    cv2.putText(frame, 'Spectrogram', (w-display_width-10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            except Exception as e:
                pass  # 忽略错误，继续显示视频
        
        return frame
    
    def run(self):
        """运行主程序"""
        # 创建OpenCV窗口并设置鼠标回调
        cv2.namedWindow('Video FFT Analyzer')
        cv2.setMouseCallback('Video FFT Analyzer', self.mouse_callback)
        
        # 启动视频捕获线程
        capture_thread = threading.Thread(target=self.capture_frames)
        capture_thread.daemon = True
        capture_thread.start()
        
        try:
            while self.running:
                with self.lock:
                    if self.display_frame is not None:
                        # 绘制频谱图
                        display_frame = self.draw_spectrogram_opencv(self.display_frame.copy())
                        
                        # 显示FPS
                        cv2.putText(display_frame, f'FPS: {self.fps:.1f}', (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # 获取帧高度
                        h = display_frame.shape[0]
                        
                        # 显示指令
                        cv2.putText(display_frame, 'Click to select ROI center', (10, h-20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        # 显示帧
                        cv2.imshow('Video FFT Analyzer', display_frame)
                
                # 检查按键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"程序运行出错: {e}")
        finally:
            self.running = False
            capture_thread.join()
            self.cap.release()
            cv2.destroyAllWindows()

def main():
    """主函数"""
    try:
        analyzer = VideoFFTAnalyzer(roi_radius=30, buffer_size=200)
        print("视频FFT分析器已启动")
        print("操作说明:")
        print("1. 在窗口中点击任意位置设置ROI中心点")
        print("2. 按 'q' 键退出程序")
        analyzer.run()
    except Exception as e:
        print(f"程序运行出错: {e}")

if __name__ == "__main__":
    main()