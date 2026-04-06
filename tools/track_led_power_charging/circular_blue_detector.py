import cv2
import numpy as np

def detect_circular_blue_regions():
    """
    检测摄像头画面中接近圆形的蓝色区域，用于检测蓝色LED
    """
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    
    # 设置默认蓝色范围 (BGR格式)
    lower_blue = np.array([100, 0, 0])
    upper_blue = np.array([255, 100, 100])
    
    print("按 'q' 退出程序")
    
    while True:
        # 获取每一帧
        ret, frame = cap.read()
        if not ret:
            break
            
        # 在原始帧上绘制结果
        display_frame = frame.copy()
        
        # 创建蓝色掩码
        blue_mask = cv2.inRange(frame, lower_blue, upper_blue)
        
        # 对掩码进行形态学操作，去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 存储符合条件的圆形区域
        circular_blue_regions = []
        
        # 遍历所有轮廓
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 过滤太小的区域
            if area > 50:
                # 计算轮廓的周长
                perimeter = cv2.arcLength(contour, True)
                
                # 计算圆形度 (circularity)
                # 圆形度 = 4π * 面积 / (周长^2)
                # 完美圆形的圆形度为1.0，越接近1.0越像圆形
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # 检查是否接近圆形 (圆形度在0.7到1.3之间)
                    if 0.7 <= circularity <= 1.3:
                        # 获取边界框
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # 检查宽高比 (接近1表示更像圆形)
                        aspect_ratio = float(w) / float(h)
                        if 0.7 <= aspect_ratio <= 1.3:
                            # 计算中心点
                            center_x = x + w // 2
                            center_y = y + h // 2
                            
                            # 添加到结果列表
                            circular_blue_regions.append({
                                'center': (center_x, center_y),
                                'contour': contour,
                                'area': area,
                                'circularity': circularity,
                                'bounding_rect': (x, y, w, h)
                            })
        
        # 绘制检测到的接近圆形的蓝色区域
        for region in circular_blue_regions:
            # 绘制轮廓
            cv2.drawContours(display_frame, [region['contour']], -1, (0, 255, 0), 2)
            
            # 绘制边界框
            x, y, w, h = region['bounding_rect']
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
            # 绘制中心点
            center = region['center']
            cv2.circle(display_frame, center, 5, (0, 0, 255), -1)
            
            # 显示信息
            cv2.putText(display_frame, f"C:{region['circularity']:.2f}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示结果
        cv2.imshow('Circular Blue Region Detector', display_frame)
        cv2.imshow('Blue Mask', blue_mask)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_circular_blue_regions()