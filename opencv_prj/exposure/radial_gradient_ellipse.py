import cv2
import numpy as np

def create_radial_gradient_ellipse(canvas_size=(800, 1000), 
                                 center=(500, 400),
                                 axes=(100, 60),
                                 center_brightness=255,
                                 edge_brightness=30):
    """
    创建一个具有放射性渐变的椭圆
    参数:
        canvas_size: 画布大小 (height, width)
        center: 椭圆中心点 (x, y)
        axes: 椭圆轴长 (major_axis, minor_axis)
        center_brightness: 中心亮度
        edge_brightness: 边缘亮度
    """
    # 创建画布
    canvas = np.zeros(canvas_size, dtype=np.uint8)
    
    # 创建椭圆mask
    mask = np.zeros(canvas_size, dtype=np.uint8)
    cv2.ellipse(mask,
                center,
                axes,
                0, 0, 360,
                255,
                -1)
    
    # 创建距离图（到椭圆中心的距离）
    y, x = np.ogrid[:canvas_size[0], :canvas_size[1]]
    # 归一化坐标到椭圆坐标系
    x = (x - center[0]) / axes[0]
    y = (y - center[1]) / axes[1]
    # 计算到中心的归一化距离
    dist = np.sqrt(x*x + y*y)
    
    # 创建渐变
    # 将距离映射到亮度值（距离为1时对应边缘亮度）
    gradient = np.clip(1 - dist, 0, 1)  # 将距离转换为0-1范围
    gradient = gradient * (center_brightness - edge_brightness) + edge_brightness
    gradient = gradient.astype(np.uint8)
    
    # 只保留椭圆内部的渐变
    result = np.where(mask > 0, gradient, 0)
    
    return result

def main():
    # 创建渐变椭圆
    result = create_radial_gradient_ellipse()
    
    # 显示结果
    cv2.imshow('Radial Gradient Ellipse', result)
    
    # 转换为BGR以便保存彩色图像
    result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('radial_gradient_ellipse.png', result_bgr)
    
    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 