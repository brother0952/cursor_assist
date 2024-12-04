import cv2
import numpy as np
import time
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

# 自动移动模式的状态
auto_move = False
last_move_time = 0
MOVE_INTERVAL = 100  # 移动间隔(ms)
move_direction = -1  # -1表示向左移动，1表示向右移动

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
    
    # 创建灰度图层
    gray_layer = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    
    # 根据move_count选择对应的椭圆组
    current_set_index = move_count + 15
    current_set = ellipse_sets[current_set_index]
    
    # 获取5个椭圆的参数
    centers = [current_set.center1, current_set.center2, current_set.center3, 
              current_set.center4, current_set.center5]
    axes = [current_set.axes1, current_set.axes2, current_set.axes3, 
            current_set.axes4, current_set.axes5]
    
    # 创建y, x网格
    y, x = np.ogrid[:canvas_height, :canvas_width]
    
    # 创建高度场
    height_field = np.zeros_like(gray_layer, dtype=float)
    
    # 创建最外层椭圆的mask
    outer_mask = np.zeros_like(gray_layer)
    actual_x = BASE_CENTER_X + centers[4][0]
    actual_y = BASE_CENTER_Y + centers[4][1]
    cv2.ellipse(outer_mask,
                (int(actual_x), int(actual_y)),
                (axes[4][0], axes[4][1]),
                0, 0, 360,
                1,
                -1)
    
    # 设置每个椭圆边界的亮度值
    boundary_values = [200, 180, 130, 10, 3]  # 从内到外的边界亮度值
    
    # 从内到外叠加每个椭圆的贡献
    for i in range(5):
        actual_x = BASE_CENTER_X + centers[i][0]
        actual_y = BASE_CENTER_Y + centers[i][1]
        
        # 归一化坐标到椭圆坐标系
        x_norm = (x - actual_x) / axes[i][0]
        y_norm = (y - actual_y) / axes[i][1]
        
        # 计算到中心的归一化距离
        dist = np.sqrt(x_norm * x_norm + y_norm * y_norm)
        
        # 创建高斯形状的贡献
        contribution = np.exp(-dist * dist * 2)
        contribution = contribution * (255 - boundary_values[i]) + boundary_values[i]
        
        # 叠加到高度场
        height_field = np.maximum(height_field, contribution)
    
    # 只保留最外层椭圆内部的区域
    height_field = height_field * outer_mask
    gray_layer = height_field.astype(np.uint8)
    
    # 将灰度图转换为BGR格式
    frame = cv2.cvtColor(gray_layer, cv2.COLOR_GRAY2BGR)
    
    # 绘制绘图区域边框
    cv2.rectangle(frame, 
                 (drawing_area_x, drawing_area_y), 
                 (drawing_area_x + drawing_area_width, drawing_area_y + drawing_area_height), 
                 (255, 255, 255),
                 1)
    
    # 添加状态信息
    cv2.putText(frame, 
                f"Current Index: {move_count} (Array Index: {current_set_index})", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                1)
    
    # 添加自动移动状态
    mode_text = "Auto Move: ON" if auto_move else "Auto Move: OFF"
    cv2.putText(frame, 
                mode_text, 
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                1)
    
    # 添加操作说明
    instructions = [
        "Controls:",
        "A/Left Arrow : Move Left",
        "D/Right Arrow: Move Right",
        "N           : Start Auto Move",
        "M           : Stop Auto Move",
        "Q           : Quit"
    ]
    
    y_pos = 700  # 起始y坐标
    for instruction in instructions:
        cv2.putText(frame, 
                    instruction, 
                    (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (200, 200, 200), 
                    1)
        y_pos += 25
    
    return frame

def handle_auto_move():
    global move_count, move_direction, last_move_time
    
    current_time = int(time.time() * 1000)  # 获取当前时间(ms)
    if current_time - last_move_time >= MOVE_INTERVAL:
        # 时间间隔达到，执行移动
        if move_direction == -1:  # 向左移动
            if move_count > MOVE_MIN:
                move_count -= 1
            else:
                move_direction = 1  # 达到左边界，改变方向
        else:  # 向右移动
            if move_count < MOVE_MAX:
                move_count += 1
            else:
                move_direction = -1  # 达到右边界，改变方向
        
        last_move_time = current_time

def main():
    global move_count, auto_move, move_direction
    
    while True:
        frame = draw_frame(move_count)
        cv2.imshow('Ellipse Animation', frame)
        
        # 如果在自动移动模式，处理自动移动
        if auto_move:
            handle_auto_move()
        
        key = cv2.waitKey(1)  # 减小等待时间以使动画更流畅
        
        if key == ord('q'):  # 按q退出
            break
        elif key == ord('a') or key == 81:  # 按a或左箭头键
            if not auto_move and move_count > MOVE_MIN:
                move_count -= 1
        elif key == ord('d') or key == 83:  # 按d或右箭头键
            if not auto_move and move_count < MOVE_MAX:
                move_count += 1
        elif key == ord('n'):  # 按n开启自动移动模式
            auto_move = True
            move_direction = -1  # 开始时向左移动
            move_count = 0  # 从中心开始
        elif key == ord('m'):  # 按m关闭自动移动模式
            auto_move = False
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()