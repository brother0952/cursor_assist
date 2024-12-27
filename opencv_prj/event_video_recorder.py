import cv2
import numpy as np
from collections import deque
import time
from pathlib import Path
from datetime import datetime
import threading
from queue import Queue
import mmap
import os

class EventVideoRecorder:
    def __init__(self, buffer_seconds=5, fps=30):
        self.buffer_seconds = buffer_seconds
        self.target_fps = fps
        self.frame_buffer = deque(maxlen=buffer_seconds * fps)
        self.write_queue = Queue(maxsize=1000)
        self.recording = False
        self.event_triggered = False
        self.last_event_time = 0
        
        # 创建输出目录
        self.output_dir = Path("event_videos")
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")
            
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 预分配内存缓冲区
        self.frame_size = self.frame_width * self.frame_height * 3  # BGR格式
        self.memory_buffer = []
        for _ in range(buffer_seconds * fps * 2):  # 双倍大小以确保足够
            self.memory_buffer.append(np.zeros((self.frame_height, self.frame_width, 3), 
                                             dtype=np.uint8))
        self.buffer_index = 0
        
        # 视频写入器和临时文件
        self.video_writer = None
        self.current_video_path = None
        self.temp_file = None
        
        # 帧计数和时间控制
        self.frame_count = 0
        self.start_time = time.time()
        
        # 写入线程控制
        self.write_thread = None
        self.should_stop = False
        self.write_event = threading.Event()
        
    def get_next_buffer(self):
        """获取下一个可用的内存缓冲区"""
        buffer = self.memory_buffer[self.buffer_index]
        self.buffer_index = (self.buffer_index + 1) % len(self.memory_buffer)
        return buffer
        
    def writer_thread(self):
        """异步写入线程"""
        while not self.should_stop:
            if not self.write_queue.empty():
                frame = self.write_queue.get()
                if frame is None:  # 结束信号
                    break
                if self.video_writer is not None:
                    self.video_writer.write(frame)
                    self.frame_count += 1
                    self.write_event.set()  # 通知写入完成
            else:
                time.sleep(0.001)  # 避免空转
                
    def start_new_recording(self):
        """开始新的录制"""
        if self.video_writer is not None:
            self.write_queue.put(None)
            if self.write_thread is not None:
                self.write_thread.join()
            self.video_writer.release()
            
        # 生成新的视频文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_video_path = self.output_dir / f"event_{timestamp}.mp4"
        
        # 创建临时文件
        temp_path = self.output_dir / f"temp_{timestamp}.mp4"
        
        # 使用 XVID 编码器
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(
            str(temp_path),
            fourcc,
            self.target_fps,
            (self.frame_width, self.frame_height)
        )
        
        # 写入缓冲区中的帧
        for frame in self.frame_buffer:
            buffer = self.get_next_buffer()
            np.copyto(buffer, frame)
            self.write_queue.put(buffer)
            
        # 启动写入线程
        self.write_thread = threading.Thread(target=self.writer_thread)
        self.write_thread.start()
            
        self.recording = True
        self.frame_count = len(self.frame_buffer)
        self.temp_file = temp_path
        print(f"开始录制: {self.current_video_path}")
        
    def stop_recording(self):
        """停止录制"""
        if self.video_writer is not None:
            self.write_queue.put(None)
            if self.write_thread is not None:
                self.write_thread.join()
            self.video_writer.release()
            self.video_writer = None
            
            # 重命名临时文件
            if self.temp_file and self.temp_file.exists():
                self.temp_file.rename(self.current_video_path)
                self.temp_file = None
                
            self.recording = False
            duration = self.frame_count / self.target_fps
            print(f"录制完成: {self.current_video_path}")
            print(f"总帧数: {self.frame_count}, 预期时长: {duration:.2f}秒")
            self.frame_count = 0
            
    def process_frame(self, frame):
        """处理每一帧"""
        # 使用预分配的缓冲区
        buffer = self.get_next_buffer()
        np.copyto(buffer, frame)
        
        # 添加到缓冲区
        self.frame_buffer.append(buffer)
        
        # 如果正在录制，将帧加入写入队列
        if self.recording:
            try:
                self.write_queue.put_nowait(buffer)
            except:
                print("写入队列已满，丢弃一帧")
            
        # 在画面上显示录制状态和时间信息
        if self.recording:
            elapsed = time.time() - self.last_event_time
            remaining = self.buffer_seconds - elapsed
            cv2.putText(frame, f"Recording ({remaining:.1f}s)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        # 显示缓冲区大小和实时帧率
        current_fps = self.frame_count / (time.time() - self.start_time) if self.recording else 0
        status_text = [
            f"Buffer: {len(self.frame_buffer)}/{self.frame_buffer.maxlen}",
            f"Queue: {self.write_queue.qsize()}/{self.write_queue.maxsize}",
            f"FPS: {current_fps:.1f}",
            f"Frames: {self.frame_count}"
        ]
        
        for i, text in enumerate(status_text):
            cv2.putText(frame, text, (10, 60 + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
        
    def run(self):
        """主循环"""
        print("按空格键触发事件，按'q'退出")
        
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
                cv2.imshow('Event Video Recorder', frame)
                
                # 检查键盘事件
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):  # 空格键触发事件
                    self.event_triggered = True
                    self.last_event_time = time.time()
                    self.start_time = time.time()
                    
                    if not self.recording:
                        self.start_new_recording()
                        
                # 检查是否需要停止录制
                if self.recording and time.time() - self.last_event_time > self.buffer_seconds:
                    self.stop_recording()
                    
        finally:
            # 清理资源
            self.should_stop = True
            self.stop_recording()
            if self.temp_file and self.temp_file.exists():
                try:
                    self.temp_file.unlink()
                except:
                    pass
            self.cap.release()
            cv2.destroyAllWindows()

def main():
    try:
        recorder = EventVideoRecorder(buffer_seconds=5, fps=30)
        recorder.run()
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main() 