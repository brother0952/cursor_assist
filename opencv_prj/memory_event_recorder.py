import cv2
import numpy as np
import time
from pathlib import Path
from datetime import datetime
import threading
from queue import Queue
import tempfile
import os
import shutil
import platform
import mmap
import io

class MemoryEventRecorder:
    def __init__(self, buffer_seconds=5, fps=30):
        self.buffer_seconds = buffer_seconds
        self.target_fps = fps
        self.recording = False
        self.event_triggered = False
        self.last_event_time = 0
        
        # 检测操作系统
        self.is_windows = platform.system() == 'Windows'
        
        # 创建输出目录
        self.output_dir = Path("event_videos")
        self.output_dir.mkdir(exist_ok=True)
        
        # 根据操作系统选择存储方式
        if self.is_windows:
            # Windows下使用内存映射文件
            self.memory_file = tempfile.SpooledTemporaryFile(max_size=1024*1024*100)  # 100MB缓存
            self.ram_dir = None
        else:
            # Linux下使用/dev/shm
            self.memory_file = None
            self.ram_dir = Path("/dev/shm/event_recorder")
            self.ram_dir.mkdir(exist_ok=True)
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")
            
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 视频写入器
        self.video_writer = None
        self.current_video_path = None
        
        # 帧计数和时间控制
        self.frame_count = 0
        self.start_time = time.time()
        
        # 状态控制
        self.recording_state = "STANDBY"  # STANDBY, RECORDING, SAVING, CLEARING
        
    def start_new_recording(self):
        """开始新的录制"""
        if self.video_writer is not None:
            self.video_writer.release()
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.is_windows:
            # Windows下重置内存文件指针
            self.memory_file.seek(0)
            self.memory_file.truncate()
            # 使用内存文件作为视频写入目标
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(
                'cache.mp4',  # 临时文件名
                fourcc,
                self.target_fps,
                (self.frame_width, self.frame_height),
                isColor=True
            )
        else:
            # Linux下使用/dev/shm
            self.current_video_path = self.ram_dir / f"event_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(
                str(self.current_video_path),
                fourcc,
                self.target_fps,
                (self.frame_width, self.frame_height)
            )
        
        self.recording = True
        self.frame_count = 0
        print(f"开始录制到内存")
        
    def save_to_disk(self):
        """将视频从内存保存到磁盘"""
        if self.is_windows:
            if self.memory_file:
                # 将内存文件内容写入磁盘
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                disk_path = self.output_dir / f"event_{timestamp}.mp4"
                self.memory_file.seek(0)
                with open(disk_path, 'wb') as f:
                    shutil.copyfileobj(self.memory_file, f)
                print(f"视频已保存到磁盘: {disk_path}")
                return True
        else:
            if self.current_video_path and self.current_video_path.exists():
                disk_path = self.output_dir / self.current_video_path.name
                shutil.copy2(str(self.current_video_path), str(disk_path))
                print(f"视频已保存到磁盘: {disk_path}")
                return True
        return False
    
    def clear_memory(self):
        """清除内存中的视频"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            
        if self.is_windows:
            if self.memory_file:
                self.memory_file.seek(0)
                self.memory_file.truncate()
        else:
            if self.current_video_path and self.current_video_path.exists():
                self.current_video_path.unlink()
        
        print("内存中的视频已清除")
            
    def process_frame(self, frame):
        """处理每一帧"""
        # 如果正在录制，写入帧
        if self.recording and self.video_writer is not None:
            if self.is_windows:
                # Windows下写入内存文件
                self.video_writer.write(frame)
                self.memory_file.write(frame.tobytes())
            else:
                # Linux下直接写入/dev/shm
                self.video_writer.write(frame)
            self.frame_count += 1
            
        # 在画面上显示状态信息
        status_text = [
            f"State: {self.recording_state}",
            f"FPS: {self.frame_count/(time.time()-self.start_time):.1f}" if self.recording else "FPS: 0.0",
            f"Frames: {self.frame_count}",
            f"System: {'Windows' if self.is_windows else 'Linux'}"
        ]
        
        for i, text in enumerate(status_text):
            cv2.putText(frame, text, (10, 30 + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
        
    def run(self):
        """主循环"""
        print("按键说明:")
        print("r: 开始录制")
        print("s: 保存到磁盘")
        print("c: 清除内存")
        print("q: 退出")
        
        frame_time = 1.0 / self.target_fps
        last_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                # 帧率控制
                current_time = time.time()
                elapsed = current_time - last_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
                last_time = current_time
                
                # 处理帧
                frame = self.process_frame(frame)
                
                # 显示画面
                cv2.imshow('Memory Event Recorder', frame)
                
                # 检查键盘事件
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('r'):  # 开始录制
                    self.recording_state = "RECORDING"
                    self.start_new_recording()
                    self.start_time = time.time()
                elif key == ord('s'):  # 保存到磁盘
                    self.recording_state = "SAVING"
                    if self.save_to_disk():
                        self.recording_state = "STANDBY"
                elif key == ord('c'):  # 清除内存
                    self.recording_state = "CLEARING"
                    self.clear_memory()
                    self.recording_state = "STANDBY"
                    
        finally:
            # 清理资源
            if self.video_writer is not None:
                self.video_writer.release()
            self.cap.release()
            cv2.destroyAllWindows()
            
            # 清理内存资源
            if self.is_windows:
                if self.memory_file:
                    self.memory_file.close()
            else:
                if self.ram_dir and self.ram_dir.exists():
                    shutil.rmtree(str(self.ram_dir))

def main():
    try:
        recorder = MemoryEventRecorder(buffer_seconds=5, fps=30)
        recorder.run()
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main() 