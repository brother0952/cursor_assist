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

def create_irregular_radial_gradient(width, height, ellipse_params):
    """
    创建不规则椭圆形径向渐变图像
    """
    # 创建坐标网格
    y, x = np.ogrid[:height, :width]
    
    # 创建输出图像
    height_field = np.zeros((height, width), dtype=float)
    
    # 创建最外层椭圆的mask
    outer_mask = np.zeros((height, width), dtype=np.uint8)
    
    # 从内到外处理每个椭圆
    for params in ellipse_params:
        center_x = params['center'][0]
        center_y = params['center'][1]
        
        # 分别计算左右两边到中心的距离
        x_dist = x - center_x
        y_dist = y - center_y
        
        # 根据点在椭圆左右两侧选择不同的x轴半径
        x_radius = np.where(x_dist < 0, 
                           params['axes_left'], 
                           params['axes_right'])
        
        # 计算归一化距离
        x_norm = x_dist / x_radius
        y_norm = y_dist / params['axes_y']
        
        # 计算到中心的归一化距离
        dist = np.sqrt(x_norm * x_norm + y_norm * y_norm)
        
        # 创建高斯形状的贡献
        contribution = np.exp(-dist * dist * 2)
        contribution = contribution * params['value']
        
        # 更新高度场
        height_field = np.maximum(height_field, contribution)
        
        # 如果是最外层椭圆，创建mask
        if params == ellipse_params[-1]:
            mask = (dist <= 1).astype(np.uint8)
            outer_mask = mask
    
    # 应用外层椭圆mask
    height_field = height_field * outer_mask
    
    return height_field.astype(np.uint8)

def draw_frame(move_count):
    # 创建黑色画布
    frame = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # 根据move_count选择对应的椭圆组
    current_set_index = move_count + 15
    current_set = ellipse_sets[current_set_index]
    
    # 构建椭圆参数列表
    ellipse_params = [
        {
            'center': (BASE_CENTER_X + current_set.center1[0], 
                      BASE_CENTER_Y + current_set.center1[1]),
            'axes_left': current_set.axes1[0],
            'axes_right': current_set.axes1[0],
            'axes_y': current_set.axes1[1],
            'value': 200
        },
        {
            'center': (BASE_CENTER_X + current_set.center2[0], 
                      BASE_CENTER_Y + current_set.center2[1]),
            'axes_left': current_set.axes2[0],
            'axes_right': current_set.axes2[0],
            'axes_y': current_set.axes2[1],
            'value': 180
        },
        {
            'center': (BASE_CENTER_X + current_set.center3[0], 
                      BASE_CENTER_Y + current_set.center3[1]),
            'axes_left': current_set.axes3[0],
            'axes_right': current_set.axes3[0],
            'axes_y': current_set.axes3[1],
            'value': 130
        },
        {
            'center': (BASE_CENTER_X + current_set.center4[0], 
                      BASE_CENTER_Y + current_set.center4[1]),
            'axes_left': current_set.axes4[0],
            'axes_right': current_set.axes4[0],
            'axes_y': current_set.axes4[1],
            'value': 10
        },
        {
            'center': (BASE_CENTER_X + current_set.center5[0], 
                      BASE_CENTER_Y + current_set.center5[1]),
            'axes_left': current_set.axes5[0],
            'axes_right': current_set.axes5[0],
            'axes_y': current_set.axes5[1],
            'value': 3
        }
    ]
    
    # 创建灰度渐变图像
    gray_layer = create_irregular_radial_gradient(
        canvas_width, 
        canvas_height, 
        ellipse_params
    )
    
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
    
    y_pos = 700
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
    
    current_time = int(time.time() * 1000)
    if current_time - last_move_time >= MOVE_INTERVAL:
        if move_direction == -1:
            if move_count > MOVE_MIN:
                move_count -= 1
            else:
                move_direction = 1
        else:
            if move_count < MOVE_MAX:
                move_count += 1
            else:
                move_direction = -1
        
        last_move_time = current_time

def main():
    global move_count, auto_move, move_direction
    
    while True:
        frame = draw_frame(move_count)
        cv2.imshow('Ellipse Animation', frame)
        
        if auto_move:
            handle_auto_move()
        
        key = cv2.waitKey(1)
        
        if key == ord('q'):
            break
        elif key == ord('a') or key == 81:
            if not auto_move and move_count > MOVE_MIN:
                move_count -= 1
        elif key == ord('d') or key == 83:
            if not auto_move and move_count < MOVE_MAX:
                move_count += 1
        elif key == ord('n'):
            auto_move = True
            move_direction = -1
            move_count = 0
        elif key == ord('m'):
            auto_move = False
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()