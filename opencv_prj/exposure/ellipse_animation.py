import cv2
import numpy as np
from ellipse_data import EllipseSet, ellipse_data

# 设置画布大小
canvas_width = 1000
canvas_height = 800
drawing_area_x = 300
drawing_area_y = 200
drawing_area_width = 320
drawing_area_height = 80

def draw_frame():
    # 创建黑色画布
    frame = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # 绘制绘图区域边框
    cv2.rectangle(frame, 
                 (drawing_area_x, drawing_area_y), 
                 (drawing_area_x + drawing_area_width, drawing_area_y + drawing_area_height), 
                 (255, 255, 255), 
                 1)
    
    # 绘制所有椭圆
    for i in range(5):
        cv2.ellipse(frame, 
                   (int(ellipse_data.centers[i][0]), int(ellipse_data.centers[i][1])),
                   ellipse_data.axes[i],
                   0,  # 角度
                   0,  # 起始角
                   360,  # 结束角
                   ellipse_data.colors[i],
                   1)  # 线宽
    
    return frame

def main():
    while True:
        frame = draw_frame()
        cv2.imshow('Ellipse Animation', frame)
        
        key = cv2.waitKey(30)
        if key == ord('q'):  # 按q退出
            break
        elif key == ord('a') or key == 81:  # 按a或左箭头键
            # 移动除了最大椭圆以外的所有椭圆
            for i in range(4):  # 只移动前4个椭圆
                x, y = ellipse_data.centers[i]
                ellipse_data.centers[i] = (x - 3, y)
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()