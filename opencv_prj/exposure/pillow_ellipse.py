from PIL import Image, ImageDraw

def create_elliptical_gradient(width, height, color1, color2):
    # 创建一个新的图像
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)

    # 计算椭圆的中心和半径
    center_x, center_y = width // 2, height // 2
    radius_x, radius_y = width // 2, height // 2

    # 从外到内绘制椭圆
    steps = min(radius_x, radius_y)  # 使用较小的半径作为步进次数
    for i in range(steps):
        # 计算当前渐变颜色
        ratio = i / steps
        r = int(color1[0] + (color2[0] - color1[0]) * ratio)
        g = int(color1[1] + (color2[1] - color1[1]) * ratio)
        b = int(color1[2] + (color2[2] - color1[2]) * ratio)
        color = (r, g, b)

        # 计算当前椭圆的边界框
        left = i
        top = i
        right = width - i
        bottom = height - i

        # 绘制椭圆
        draw.ellipse(
            [left, top, right - 1, bottom - 1],  # 减1避免边界问题
            fill=color
        )

    return image

# 使用示例
gradient_image = create_elliptical_gradient(400, 300, (255, 0, 0), (0, 0, 255))
gradient_image.show()