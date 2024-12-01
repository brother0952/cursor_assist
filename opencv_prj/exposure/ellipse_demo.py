import cv2
import numpy as np
import configparser

def load_config():
    config = configparser.ConfigParser()
    config.read('ellipse_config.ini', encoding='utf-8')
    
    return {
        'center_x': config.getint('Ellipse', 'center_x'),
        'center_y': config.getint('Ellipse', 'center_y'),
        'major_axis': config.getint('Ellipse', 'major_axis'),
        'minor_axis': config.getint('Ellipse', 'minor_axis'),
        'angle': config.getint('Ellipse', 'angle'),
        'move_step': config.getint('Ellipse', 'move_step'),
        'perspective_tl_x': config.getint('Ellipse', 'perspective_tl_x'),
        'perspective_tl_y': config.getint('Ellipse', 'perspective_tl_y'),
        'perspective_tr_x': config.getint('Ellipse', 'perspective_tr_x'),
        'perspective_tr_y': config.getint('Ellipse', 'perspective_tr_y'),
        'perspective_bl_x': config.getint('Ellipse', 'perspective_bl_x'),
        'perspective_bl_y': config.getint('Ellipse', 'perspective_bl_y'),
        'perspective_br_x': config.getint('Ellipse', 'perspective_br_x'),
        'perspective_br_y': config.getint('Ellipse', 'perspective_br_y')
    }

def apply_perspective(img, corners):
    rows, cols = img.shape[:2]
    
    # 创建源点（原始图像的四个角点）
    src_points = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
    
    # 创建目标点（变换后的四个角点）
    tl = [corners['tl_x'], corners['tl_y']]  # 左上
    tr = [cols-1 + corners['tr_x'], corners['tr_y']]  # 右上
    bl = [corners['bl_x'], rows-1 + corners['bl_y']]  # 左下
    br = [cols-1 + corners['br_x'], rows-1 + corners['br_y']]  # 右下
    
    dst_points = np.float32([tl, tr, bl, br])
    
    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # 应用透视变换
    result = cv2.warpPerspective(img, matrix, (cols, rows))
    return result

def draw_control_points(img, corners, current_corner):
    rows, cols = img.shape[:2]
    
    # 定义控制点的位置
    points = {
        'tl': (corners['tl_x'], corners['tl_y']),  # 左上
        'tr': (cols-1 + corners['tr_x'], corners['tr_y']),  # 右上
        'bl': (corners['bl_x'], rows-1 + corners['bl_y']),  # 左下
        'br': (cols-1 + corners['br_x'], rows-1 + corners['br_y'])  # 右下
    }
    
    # 绘制控制点和连线
    for name, point in points.items():
        color = (0, 0, 255) if name == current_corner else (255, 0, 0)  # 当前选中点为红色，其他为蓝色
        cv2.circle(img, (int(point[0]), int(point[1])), 3, color, -1)
        cv2.putText(img, name.upper(), (int(point[0])-10, int(point[1])-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # 绘制控制点之间的连线
    cv2.line(img, (int(points['tl'][0]), int(points['tl'][1])), 
             (int(points['tr'][0]), int(points['tr'][1])), (255, 0, 0), 1)
    cv2.line(img, (int(points['tr'][0]), int(points['tr'][1])), 
             (int(points['br'][0]), int(points['br'][1])), (255, 0, 0), 1)
    cv2.line(img, (int(points['br'][0]), int(points['br'][1])), 
             (int(points['bl'][0]), int(points['bl'][1])), (255, 0, 0), 1)
    cv2.line(img, (int(points['bl'][0]), int(points['bl'][1])), 
             (int(points['tl'][0]), int(points['tl'][1])), (255, 0, 0), 1)

def main():
    # 创建画布
    canvas_width = 320
    canvas_height = 80
    
    # 加载配置
    config = load_config()
    center_x = config['center_x']
    major_axis = config['major_axis']
    minor_axis = config['minor_axis']
    
    # 初始化四个角点的透视参数
    corners = {
        'tl_x': config['perspective_tl_x'],
        'tl_y': config['perspective_tl_y'],
        'tr_x': config['perspective_tr_x'],
        'tr_y': config['perspective_tr_y'],
        'bl_x': config['perspective_bl_x'],
        'bl_y': config['perspective_bl_y'],
        'br_x': config['perspective_br_x'],
        'br_y': config['perspective_br_y']
    }
    
    # 当前选中的控制点
    current_corner = 'tl'
    
    while True:
        # 创建黑色画布
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # 绘制椭圆
        center = (center_x, config['center_y'])
        axes = (major_axis, minor_axis)
        cv2.ellipse(canvas, center, axes, config['angle'], 
                    0, 360, (0, 255, 0), -1)
        
        cv2.ellipse(canvas, (50,50), (major_axis-5, minor_axis), config['angle'], 
                    0, 360, (0, 255, 0), -1)
        
        # 应用透视变换
        # canvas = apply_perspective(canvas, corners)
        
        # 绘制控制点和连线
        # draw_control_points(canvas, corners, current_corner)
        
        # 显示参数信息
        info_text = [
            f'Pos:({center_x},{config["center_y"]})',
            f'Current:{current_corner}',
            f'TL:({corners["tl_x"]},{corners["tl_y"]})',
            f'TR:({corners["tr_x"]},{corners["tr_y"]})',
            f'BL:({corners["bl_x"]},{corners["bl_y"]})',
            f'BR:({corners["br_x"]},{corners["br_y"]})'
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(canvas, text, (10, 15 + i*15), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.4, (255, 255, 255), 1)
        
        # 显示画布
        cv2.imshow('Ellipse Demo', canvas)
        
        # 等待按键
        key = cv2.waitKey(30) & 0xFF
        
        # 处理按键
        if key == ord('q'):  # 按q退出
            break
        elif key == ord('a'):  # 按a向左移动
            center_x = max(0, center_x - config['move_step'])
        elif key == ord('d'):  # 按d向右移动
            center_x = min(canvas_width, center_x + config['move_step'])
            
        # 选择控制点
        elif key == ord('1'):  # 选择左上角
            current_corner = 'tl'
        elif key == ord('2'):  # 选择右上角
            current_corner = 'tr'
        elif key == ord('3'):  # 选择左下角
            current_corner = 'bl'
        elif key == ord('4'):  # 选择右下角
            current_corner = 'br'
            
        # 移动当前控制点
        elif key == ord('i'):  # 向上移动
            corners[f'{current_corner}_y'] = max(-40, corners[f'{current_corner}_y'] - 2)
        elif key == ord('k'):  # 向下移动
            corners[f'{current_corner}_y'] = min(40, corners[f'{current_corner}_y'] + 2)
        elif key == ord('j'):  # 向左移动
            corners[f'{current_corner}_x'] = max(-40, corners[f'{current_corner}_x'] - 2)
        elif key == ord('l'):  # 向右移动
            corners[f'{current_corner}_x'] = min(40, corners[f'{current_corner}_x'] + 2)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 