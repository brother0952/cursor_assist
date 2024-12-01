import cv2
import numpy as np

def draw_transformed_ellipse(canvas, center, axes, perspective_points, angle=0, color=(0, 255, 0), thickness=-1, scale=1.0):
    """
    在画布上绘制一个经过透视变换的椭圆
    
    参数:
    canvas: ndarray, 输入画布
    center: tuple, 椭圆中心点坐标 (x, y)
    axes: tuple, 椭圆的长轴和短轴 (major_axis, minor_axis)
    perspective_points: dict, 透视变换的四个控制点
        格式: {
            'tl': (x, y),  # 左上角
            'tr': (x, y),  # 右上角
            'bl': (x, y),  # 左下角
            'br': (x, y)   # 右下角
        }
    angle: float, 椭圆的旋转角度（度）
    color: tuple, 椭圆的颜色，BGR格式
    thickness: int, 线条粗细，-1表示填充
    scale: float, 显示时的缩放比例，不影响实际图像大小
    
    返回:
    ndarray: 变换后的画布
    """
    # 创建临时画布
    temp_canvas = np.zeros_like(canvas)
    
    # 绘制椭圆
    cv2.ellipse(temp_canvas, center, axes, angle, 0, 360, color, thickness)
    
    # 获取画布尺寸
    rows, cols = canvas.shape[:2]
    
    # 创建源点
    src_points = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
    
    # 创建目标点
    dst_points = np.float32([
        perspective_points['tl'],
        perspective_points['tr'],
        perspective_points['bl'],
        perspective_points['br']
    ])
    
    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # 应用透视变换
    result = cv2.warpPerspective(temp_canvas, matrix, (cols, rows))
    
    # 如果需要缩放
    if scale != 1.0:
        scaled_size = (int(cols * scale), int(rows * scale))
        result = cv2.resize(result, scaled_size, interpolation=cv2.INTER_LINEAR)
    
    return result

# 使用示例：
if __name__ == "__main__":
    # 创建画布
    canvas_width = 320
    canvas_height = 80
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # 定义参数
    center = (160, 40)
    axes = (20, 12)
    per_r_offset = -20
    perspective_points = {
        'tl': (0, 0),
        'tr': (canvas_width-1, 0-per_r_offset),
        'bl': (0, canvas_height-1),
        'br': (canvas_width-1, canvas_height-1+per_r_offset)
    }
    
    # 绘制变换后的椭圆（放大2倍显示）
    result = draw_transformed_ellipse(canvas, center, axes, perspective_points, scale=5.0)
    result = draw_transformed_ellipse(result, (50,40), (14,12), perspective_points, scale=5.0)
    
    # 显示结果
    cv2.imshow('Transformed Ellipse', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 