import cv2
import numpy as np
from pathlib import Path
import time
from datetime import datetime
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from opencv_prj.utils.plot_utils import setup_chinese_font

class SunsetAnalyzer:
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.output_dir = Path("sunset_analysis")
        self.output_dir.mkdir(exist_ok=True)
        
        # 光流参数
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # 特征点检测参数
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # 分析结果
        self.brightness_history = []
        self.motion_history = []
        self.timestamps = []
        
    def open_video(self, video_path: str) -> bool:
        """打开视频文件"""
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return False
        return True
        
    def analyze_frame(self, frame: np.ndarray, prev_frame: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """分析单帧图像"""
        # 计算亮度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # 计算运动量
        motion = 0.0
        if prev_frame is not None:
            # 计算光流
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0
            )
            # 计算运动量（光流大小的平均值）
            motion = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
            
        return brightness, motion
        
    def process_video(self, video_path: str, sample_interval: int = 1):
        """处理视频"""
        if not self.open_video(video_path):
            return
            
        frame_count = 0
        prev_frame = None
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            if frame_count % sample_interval == 0:
                # 分析帧
                brightness, motion = self.analyze_frame(frame, prev_frame)
                
                # 记录结果
                self.brightness_history.append(brightness)
                self.motion_history.append(motion)
                self.timestamps.append(frame_count / self.cap.get(cv2.CAP_PROP_FPS))
                
                # 显示进度
                if frame_count % 100 == 0:
                    print(f"已处理 {frame_count} 帧...")
                
                prev_frame = frame.copy()
            
            frame_count += 1
            
        self.cap.release()
        
        # 生成分析报告
        self.generate_report()
        
        print(f"处理完成! 总用时: {time.time() - start_time:.2f} 秒")
        
    def generate_report(self):
        """生成分析报告"""
        # 设置中文字体
        setup_chinese_font()
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 绘制亮度变化
        ax1.plot(self.timestamps, self.brightness_history, 'b-', label='亮度')
        ax1.set_title('视频亮度变化')
        ax1.set_xlabel('时间 (秒)')
        ax1.set_ylabel('平均亮度')
        ax1.grid(True)
        ax1.legend()
        
        # 绘制运动量变化
        ax2.plot(self.timestamps, self.motion_history, 'r-', label='运动量')
        ax2.set_title('视频运动量变化')
        ax2.set_xlabel('时间 (秒)')
        ax2.set_ylabel('平均运动量')
        ax2.grid(True)
        ax2.legend()
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(self.output_dir / f"sunset_analysis_{timestamp}.png")
        plt.close()
        
        # 保存数据
        np.savez(
            self.output_dir / f"sunset_data_{timestamp}.npz",
            timestamps=self.timestamps,
            brightness=self.brightness_history,
            motion=self.motion_history
        )

def main():
    # 创建分析器实例
    analyzer = SunsetAnalyzer()
    
    # 处理视频
    video_path = input("请输入视频文件路径: ")
    analyzer.process_video(video_path)

if __name__ == "__main__":
    main() 