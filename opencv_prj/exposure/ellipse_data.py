from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class EllipseSet:
    # 5个椭圆的中心坐标
    center1: Tuple[int, int]
    center2: Tuple[int, int]
    center3: Tuple[int, int]
    center4: Tuple[int, int]
    center5: Tuple[int, int]
    
    # 5个椭圆的轴信息 (长轴,短轴,自定义参数)
    axes1: Tuple[int, int, int]
    axes2: Tuple[int, int, int]
    axes3: Tuple[int, int, int]
    axes4: Tuple[int, int, int]
    axes5: Tuple[int, int, int]

# 创建15个EllipseSet实例的数组
ellipse_sets: List[EllipseSet] = [
    # 第1组
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(30, 20, 1), axes2=(50, 30, 2), axes3=(70, 40, 3), axes4=(100, 70, 4), axes5=(130, 100, 5)
    ),
    # 第2组
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(35, 22, 1), axes2=(55, 32, 2), axes3=(75, 42, 3), axes4=(105, 72, 4), axes5=(135, 102, 5)
    ),
    # 第3组
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(40, 24, 1), axes2=(60, 34, 2), axes3=(80, 44, 3), axes4=(110, 74, 4), axes5=(140, 104, 5)
    ),
    # 第4组
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(45, 26, 1), axes2=(65, 36, 2), axes3=(85, 46, 3), axes4=(115, 76, 4), axes5=(145, 106, 5)
    ),
    # 第5组
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(50, 28, 1), axes2=(70, 38, 2), axes3=(90, 48, 3), axes4=(120, 78, 4), axes5=(150, 108, 5)
    ),
    # 第6组
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(55, 30, 1), axes2=(75, 40, 2), axes3=(95, 50, 3), axes4=(125, 80, 4), axes5=(155, 110, 5)
    ),
    # 第7组
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(60, 32, 1), axes2=(80, 42, 2), axes3=(100, 52, 3), axes4=(130, 82, 4), axes5=(160, 112, 5)
    ),
    # 第8组
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(65, 34, 1), axes2=(85, 44, 2), axes3=(105, 54, 3), axes4=(135, 84, 4), axes5=(165, 114, 5)
    ),
    # 第9组
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(70, 36, 1), axes2=(90, 46, 2), axes3=(110, 56, 3), axes4=(140, 86, 4), axes5=(170, 116, 5)
    ),
    # 第10组
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(75, 38, 1), axes2=(95, 48, 2), axes3=(115, 58, 3), axes4=(145, 88, 4), axes5=(175, 118, 5)
    ),
    # 第11组
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(80, 40, 1), axes2=(100, 50, 2), axes3=(120, 60, 3), axes4=(150, 90, 4), axes5=(180, 120, 5)
    ),
    # 第12组
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(85, 42, 1), axes2=(105, 52, 2), axes3=(125, 62, 3), axes4=(155, 92, 4), axes5=(185, 122, 5)
    ),
    # 第13组
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(90, 44, 1), axes2=(110, 54, 2), axes3=(130, 64, 3), axes4=(160, 94, 4), axes5=(190, 124, 5)
    ),
    # 第14组
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(95, 46, 1), axes2=(115, 56, 2), axes3=(135, 66, 3), axes4=(165, 96, 4), axes5=(195, 126, 5)
    ),
    # 第15组
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(100, 48, 1), axes2=(120, 58, 2), axes3=(140, 68, 3), axes4=(170, 98, 4), axes5=(200, 128, 5)
    )
]

# 访问方式：
# 第i组第j个椭圆的中心坐标: ellipse_sets[i].center{j}
# 第i组第j个椭圆的轴信息: ellipse_sets[i].axes{j}
# 第i组第j个椭圆的长轴: ellipse_sets[i].axes{j}[0]
# 第i组第j个椭圆的短轴: ellipse_sets[i].axes{j}[1]
# 第i组第j个椭圆的自定义参数: ellipse_sets[i].axes{j}[2]
