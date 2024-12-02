import cv2
import numpy as np
from ellipse_data import EllipseSet, ellipse_sets

# 设置画布大小
canvas_width = 1000
canvas_height = 800
drawing_area_x = 300
drawing_area_y = 200
drawing_area_width = 320
drawing_area_height = 80

# 移动限制和初始值
MOVE_MIN = -15
MOVE_MAX = 15
move_count = 0  # 初始位置为0，向左为负，向右为正

# 椭圆的初始中心坐标
BASE_CENTER_X = 460
BASE_CENTER_Y = 240


def special_ellipse(canvas, center:tuple, axesx_left, axesx_right,axesy, color):
    cv2.ellipse(canvas, 
                center, 
                (axesx_right, axesy), 
                180,  # 角度
                90,  # 起始角
                270,  # 结束角
                color, 
                -1)  # 填充
    
    cv2.ellipse(canvas, 
                center, 
                (axesx_left, axesy), 
                0, 
                90, 
                270, 
                color, 
                -1)

def draw_frame(move_count):
    # 创建黑色画布
    frame = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # 绘制绘图区域边框
    cv2.rectangle(frame, 
                 (drawing_area_x, drawing_area_y), 
                 (drawing_area_x + drawing_area_width, drawing_area_y + drawing_area_height), 
                 (255, 255, 255), 
                 1)
    
    # 根据move_count选择对应的椭圆组
    current_set_index = move_count + 15  # 将-15到+15映射到0到30
    current_set = ellipse_sets[current_set_index]
    
    # 绘制5个椭圆
    centers = [current_set.center1, current_set.center2, current_set.center3, 
              current_set.center4, current_set.center5]
    axes = [current_set.axes1, current_set.axes2, current_set.axes3, 
            current_set.axes4, current_set.axes5]
    
    for i in range(5):
        # 计算实际绘制位置 = 基准位置 + 相对位置
        actual_x = BASE_CENTER_X + centers[i][0]
        actual_y = BASE_CENTER_Y + centers[i][1]
        
        cv2.ellipse(frame, 
                   (int(actual_x), int(actual_y)),
                   (axes[i][0], axes[i][1]),  # 长短轴
                   0,  # 角度
                   0,  # 起始角
                   360,  # 结束角
                   (0, 0, 255),  # 红色
                   1)  # 线宽
    
    return frame

def main():
    global move_count
    
    while True:
        frame = draw_frame(move_count)
        cv2.imshow('Ellipse Animation', frame)
        
        key = cv2.waitKey(30)
        if key == ord('q'):  # 按q退出
            break
        elif key == ord('a') or key == 81:  # 按a或左箭头键
            if move_count > MOVE_MIN:  # 检查是否达到最左边界
                move_count -= 1
                current_set = ellipse_sets[move_count + 15]
                # 更新前4个椭圆的相对位置
                for i in range(4):
                    x, y = getattr(current_set, f'center{i+1}')
                    setattr(current_set, f'center{i+1}', (x - 3, y))
                
        elif key == ord('d') or key == 83:  # 按d或右箭头键
            if move_count < MOVE_MAX:  # 检查是否达到最右边界
                move_count += 1
                current_set = ellipse_sets[move_count + 15]
                # 更新前4个椭圆的相对位置
                for i in range(4):
                    x, y = getattr(current_set, f'center{i+1}')
                    setattr(current_set, f'center{i+1}', (x + 3, y))
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()