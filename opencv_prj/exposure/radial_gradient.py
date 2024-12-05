import numpy as np

def create_irregular_radial_gradient(width, height, ellipse_params):
    """
    创建不规则椭圆形径向渐变图像
    
    参数:
    width: 图像宽度
    height: 图像高度
    ellipse_params: 列表，每个元素是一个字典，包含：
        center: (x, y) 椭圆中心的偏移量
        axes_left: 左半边的长轴
        axes_right: 右半边的长轴
        axes_y: y轴半径
        value: 该椭圆的亮度值 (0~255)
    """
    # 创建坐标网格
    y, x = np.ogrid[:height, :width]
    
    # 创建输出图像
    height_field = np.zeros((height, width), dtype=float)
    
    # 创建最外层椭圆的mask
    outer_mask = np.zeros((height, width), dtype=np.uint8)
    
    # 从内到外处理每个椭圆
    for params in ellipse_params:
        center_x = width/2 + params['center'][0]
        center_y = height/2 + params['center'][1]
        
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

if __name__ == "__main__":
    # 测试代码
    from PIL import Image
    
    # 定义不规则椭圆参数
    ellipse_params = [
        {
            'center': (0, 0),          # 中心偏移量
            'axes_left': 100,          # 左半边x轴半径
            'axes_right': 120,         # 右半边x轴半径
            'axes_y': 80,              # y轴半径
            'value': 255               # 亮度值
        },
        {
            'center': (5, 2),
            'axes_left': 150,
            'axes_right': 160,
            'axes_y': 100,
            'value': 200
        },
        {
            'center': (8, 3),
            'axes_left': 180,
            'axes_right': 200,
            'axes_y': 120,
            'value': 150
        },
        {
            'center': (10, 5),
            'axes_left': 220,
            'axes_right': 250,
            'axes_y': 150,
            'value': 100
        },
        {
            'center': (12, 6),
            'axes_left': 250,
            'axes_right': 280,
            'axes_y': 170,
            'value': 50
        }
    ]
    
    # 创建渐变图像
    gradient = create_irregular_radial_gradient(
        width=800,
        height=600,
        ellipse_params=ellipse_params
    )
    
    # 转换为PIL图像并显示
    image = Image.fromarray(gradient)
    image.show()
    
    # 可选：保存图像
    # image.save('irregular_radial_gradient.png')