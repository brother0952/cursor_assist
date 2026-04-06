import cv2
import numpy as np
from collections import deque
import threading
import time

class HarmonicAnalyzer:
    def __init__(self, roi_radius=30, buffer_size=256):
        """
        初始化谐波分析器
        
        参数:
        roi_radius: ROI区域半径（像素）
        buffer_size: 用于FFT的时间缓冲区大小，应该是2的幂次以提高FFT效率
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

    def draw_frequency_spectrum(self, frame):
        """在OpenCV画面上绘制频率谱曲线"""
        if len(self.brightness_series) < 16:  # 至少需要一些数据
            return frame
            
        # 转换为numpy数组
        brightness_array = np.array(self.brightness_series)
        
        # 如果数据太少则返回
        if len(brightness_array) < 16:
            return frame
            
        # 补零到buffer_size长度
        if len(brightness_array) < self.buffer_size:
            padded_data = np.pad(brightness_array, (0, self.buffer_size - len(brightness_array)), 
                                mode='constant', constant_values=np.mean(brightness_array))
        else:
            padded_data = brightness_array
            
        # 移除直流分量（均值）
        padded_data = padded_data - np.mean(padded_data)
        
        # 应用窗函数（汉宁窗）
        windowed_data = padded_data * np.hanning(len(padded_data))
        
        # 计算FFT
        fft_result = np.fft.fft(windowed_data)
        magnitude = np.abs(fft_result)[:len(fft_result)//2]  # 取一半（正频率部分）
        frequencies = np.fft.fftfreq(len(padded_data), d=1.0/self.fps)[:len(fft_result)//2]
        
        # 创建频谱图像区域
        h, w = frame.shape[:2]
        graph_height = h // 3
        graph_width = w // 3
        graph_img = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)
        
        # 绘制频率轴和幅度轴
        if len(magnitude) > 1:
            # 归一化幅度值到图像高度
            max_magnitude = np.max(magnitude)
            if max_magnitude > 0:
                normalized_magnitudes = (magnitude / max_magnitude) * (graph_height - 20)
            else:
                normalized_magnitudes = magnitude
                
            # 绘制频谱曲线
            for i in range(len(normalized_magnitudes) - 1):
                # 计算x坐标（频率）
                x1 = int(i * graph_width / len(normalized_magnitudes))
                x2 = int((i + 1) * graph_width / len(normalized_magnitudes))
                
                # 计算y坐标（幅度）
                y1 = int(graph_height - normalized_magnitudes[i] - 10)
                y2 = int(graph_height - normalized_magnitudes[i + 1] - 10)
                
                # 确保坐标在有效范围内
                x1 = max(0, min(graph_width - 1, x1))
                x2 = max(0, min(graph_width - 1, x2))
                y1 = max(0, min(graph_height - 1, y1))
                y2 = max(0, min(graph_height - 1, y2))
                
                # 绘制线条
                cv2.line(graph_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
            # 绘制峰值点（谐波）
            for i in range(1, len(normalized_magnitudes) - 1):
                # 简单的峰值检测
                if (magnitude[i] > magnitude[i-1]) and (magnitude[i] > magnitude[i+1]) and (magnitude[i] > max_magnitude * 0.1):
                    x = int(i * graph_width / len(normalized_magnitudes))
                    y = int(graph_height - normalized_magnitudes[i] - 10)
                    x = max(0, min(graph_width - 1, x))
                    y = max(0, min(graph_height - 1, y))
                    cv2.circle(graph_img, (x, y), 3, (0, 0, 255), -1)
                    
                    # 标记频率值
                    freq_text = f"{frequencies[i]:.1f}Hz"
                    cv2.putText(graph_img, freq_text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # 绘制坐标轴
            cv2.line(graph_img, (0, graph_height-5), (graph_width-1, graph_height-5), (255, 255, 255), 1)  # X轴
            cv2.line(graph_img, (5, 0), (5, graph_height-1), (255, 255, 255), 1)  # Y轴
            
            # 添加标签
            cv2.putText(graph_img, 'Frequency Spectrum', (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 将频谱图画到主画面右上角
            frame[10:10+graph_height, w-graph_width-10:w-10] = graph_img
            
        return frame
    
    def run(self):
        """运行主程序"""
        # 创建OpenCV窗口并设置鼠标回调
        cv2.namedWindow('Harmonic Analyzer')
        cv2.setMouseCallback('Harmonic Analyzer', self.mouse_callback)
        
        # 启动视频捕获线程
        capture_thread = threading.Thread(target=self.capture_frames)
        capture_thread.daemon = True
        capture_thread.start()
        
        try:
            while self.running:
                with self.lock:
                    if self.display_frame is not None:
                        # 绘制频谱图
                        display_frame = self.draw_frequency_spectrum(self.display_frame.copy())
                        
                        # 显示FPS
                        cv2.putText(display_frame, f'FPS: {self.fps:.1f}', (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # 获取帧高度
                        h = display_frame.shape[0]
                        
                        # 显示指令
                        cv2.putText(display_frame, 'Click to select ROI center', (10, h-20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        # 显示帧
                        cv2.imshow('Harmonic Analyzer', display_frame)
                
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
        analyzer = HarmonicAnalyzer(roi_radius=30, buffer_size=256)
        print("谐波分析器已启动")
        print("操作说明:")
        print("1. 在窗口中点击任意位置设置ROI中心点")
        print("2. 按 'q' 键退出程序")
        analyzer.run()
    except Exception as e:
        print(f"程序运行出错: {e}")

if __name__ == "__main__":
    main()