import cv2
import numpy as np

def create_two_ellipse_gradient(canvas_size=(800, 1000),
                              center=(500, 400),
                              inner_axes=(80, 50),
                              outer_axes=(120, 75),
                              inner_brightness=200,
                              outer_brightness=50):
    """
    创建两个椭圆之间的放射性渐变
    """
    # 创建画布
    canvas = np.zeros(canvas_size, dtype=np.uint8)
    
    # 创建两个椭圆的mask
    inner_mask = np.zeros(canvas_size, dtype=np.uint8)
    outer_mask = np.zeros(canvas_size, dtype=np.uint8)
    
    # 绘制两个椭圆
    cv2.ellipse(inner_mask, center, inner_axes, 0, 0, 360, 1, -1)
    cv2.ellipse(outer_mask, center, outer_axes, 0, 0, 360, 1, -1)
    
    # 获取两个椭圆之间的区域
    ring_mask = outer_mask - inner_mask
    
    # 创建距离场
    y, x = np.ogrid[:canvas_size[0], :canvas_size[1]]
    x = x - center[0]
    y = y - center[1]
    
    # 计算每个点到中心的实际距离
    dist = np.sqrt(x*x + y*y)
    
    # 计算内外椭圆的平均半径
    inner_radius = np.mean(inner_axes)
    outer_radius = np.mean(outer_axes)
    
    # 创建渐变
    # 将距离归一化到0-1范围
    ratio = (dist - inner_radius) / (outer_radius - inner_radius)
    ratio = np.clip(ratio, 0, 1)
    
    # 映射到亮度值范围
    gradient = inner_brightness * (1 - ratio) + outer_brightness * ratio
    
    # 应用mask
    result = np.zeros_like(canvas)
    result[ring_mask > 0] = gradient[ring_mask > 0]
    
    return result

def main():
    # 创建渐变椭圆
    result = create_two_ellipse_gradient(
        canvas_size=(800, 1000),
        center=(500, 400),
        inner_axes=(100, 60),  # 调整椭圆大小
        outer_axes=(150, 90),
        inner_brightness=200,
        outer_brightness=50
    )
    
    # 显示结果
    cv2.imshow('Two Ellipse Gradient', result)
    cv2.imwrite('two_ellipse_gradient.png', result)
    
    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 