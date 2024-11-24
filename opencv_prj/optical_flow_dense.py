import cv2
import numpy as np

def dense_optical_flow_demo():
    """稠密光流演示 - Farneback方法"""
    # 打开视频
    cap = cv2.VideoCapture(0)  # 使用摄像头，也可以改为视频文件路径
    
    # 读取第一帧
    ret, old_frame = cap.read()
    if not ret:
        print("无法读取视频源")
        return
    
    # 转换为灰度图
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # 创建HSV图像，用于可视化
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255  # 设置饱和度为最大
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算稠密光流
        flow = cv2.calcOpticalFlowFarneback(
            old_gray, 
            frame_gray, 
            None,
            pyr_scale=0.5,  # 金字塔缩放比例
            levels=3,       # 金字塔层数
            winsize=15,     # 窗口大小
            iterations=3,   # 迭代次数
            poly_n=5,      # 多项式展开的邻域大小
            poly_sigma=1.2, # 高斯标准差
            flags=0
        )
        
        # 计算光流的大小和方向
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # 将方向转换为色调
        hsv[..., 0] = ang * 180 / np.pi / 2
        # 将大小转换为亮度
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        # 转换回BGR以显示
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 显示结果
        cv2.imshow('Dense Optical Flow', rgb)
        cv2.imshow('Original', frame)
        
        # 更新前一帧
        old_gray = frame_gray.copy()
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    dense_optical_flow_demo() 