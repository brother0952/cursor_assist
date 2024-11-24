import cv2
import numpy as np
from opencv_prj.utils.plot_utils import setup_chinese_font
setup_chinese_font()

class OpticalFlowDemo:
    def __init__(self):
        self.cap = None
        self.flow_type = 'sparse'  # 'sparse' or 'dense'
        self.tracking = True
        
    def setup_camera(self):
        """设置摄像头"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("无法打开摄像头")
            return False
        return True
        
    def process_dense_flow(self, old_gray, frame_gray):
        """处理稠密光流"""
        flow = cv2.calcOpticalFlowFarneback(
            old_gray, frame_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )
        
        # 计算大小和方向
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # 创建HSV图像
        hsv = np.zeros((frame_gray.shape[0], frame_gray.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
    def process_sparse_flow(self, old_gray, frame_gray, p0, mask):
        """处理稀疏光流"""
        if p0 is None or len(p0) == 0:
            return None, None, mask
            
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
        
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
            
            # 绘制轨迹
            for new, old in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), 
                              (0, 255, 0), 2)
            
            return good_new.reshape(-1, 1, 2), mask
            
        return None, mask
        
    def run(self):
        """运行演示"""
        if not self.setup_camera():
            return
            
        ret, old_frame = self.cap.read()
        if not ret:
            return
            
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(old_frame)
        
        # 特征点检测参数
        feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.tracking:
                if self.flow_type == 'dense':
                    # 稠密光流
                    flow_vis = self.process_dense_flow(old_gray, frame_gray)
                    cv2.imshow('Optical Flow', flow_vis)
                else:
                    # 稀疏光流
                    p0, mask = self.process_sparse_flow(old_gray, frame_gray, p0, mask)
                    if p0 is not None:
                        img = cv2.add(frame, mask)
                        cv2.imshow('Optical Flow', img)
                        
                        # 重新检测特征点
                        if len(p0) < 50:
                            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, 
                                                       **feature_params)
                            mask = np.zeros_like(frame)
            
            # 显示原始帧
            cv2.imshow('Original', frame)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.flow_type = 'sparse'
                mask = np.zeros_like(frame)
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            elif key == ord('d'):
                self.flow_type = 'dense'
            elif key == ord('t'):
                self.tracking = not self.tracking
                if self.tracking:
                    mask = np.zeros_like(frame)
                    p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            
            old_gray = frame_gray.copy()
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = OpticalFlowDemo()
    demo.run() 