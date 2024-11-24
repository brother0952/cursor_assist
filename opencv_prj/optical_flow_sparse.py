import cv2
import numpy as np

def sparse_optical_flow_demo():
    """稀疏光流演示 - Lucas-Kanade方法"""
    # 打开视频
    cap = cv2.VideoCapture(0)  # 使用摄像头，也可以改为视频文件路径
    
    # 读取第一帧
    ret, old_frame = cap.read()
    if not ret:
        print("无法读取视频源")
        return
        
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # 创建随机点
    feature_params = dict(
        maxCorners=100,      # 最大角点数
        qualityLevel=0.3,    # 质量水平
        minDistance=7,       # 最小距离
        blockSize=7          # 块大小
    )
    
    # 使用Shi-Tomasi角点检测
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    
    # 创建绘制用的遮罩
    mask = np.zeros_like(old_frame)
    
    # 光流参数
    lk_params = dict(
        winSize=(15, 15),     # 搜索窗口大小
        maxLevel=2,           # 金字塔层数
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # 随机颜色
    color = np.random.randint(0, 255, (100, 3))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if p0 is not None and len(p0) > 0:
            # 计算光流
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                old_gray, 
                frame_gray, 
                p0, 
                None, 
                **lk_params
            )
            
            if p1 is not None:
                # 选择好的点
                good_new = p1[st==1]
                good_old = p0[st==1]
                
                # 绘制轨迹
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), 
                                  color[i].tolist(), 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 5, 
                                     color[i].tolist(), -1)
                
                # 更新点集
                p0 = good_new.reshape(-1, 1, 2)
        
        # 显示结果
        img = cv2.add(frame, mask)
        cv2.imshow('Sparse Optical Flow', img)
        
        # 更新前一帧
        old_gray = frame_gray.copy()
        
        # 每隔一段时间重新检测特征点
        if len(p0) < 50:  # 如果跟踪的点太少
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            mask = np.zeros_like(old_frame)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sparse_optical_flow_demo() 