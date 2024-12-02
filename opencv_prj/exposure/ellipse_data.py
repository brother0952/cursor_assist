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

# 创建31个EllipseSet实例的数组（索引0-30对应位移-15到+15）
ellipse_sets: List[EllipseSet] = [
    # 索引0
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(15, 5, 1), axes2=(35, 15, 2), axes3=(55, 25, 3), axes4=(85, 55, 4), axes5=(115, 85, 5)
    ),
    # 索引1
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(16, 6, 1), axes2=(36, 16, 2), axes3=(56, 26, 3), axes4=(86, 56, 4), axes5=(116, 86, 5)
    ),
    # 索引2
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(17, 7, 1), axes2=(37, 17, 2), axes3=(57, 27, 3), axes4=(87, 57, 4), axes5=(117, 87, 5)
    ),
    # 索引3
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(18, 8, 1), axes2=(38, 18, 2), axes3=(58, 28, 3), axes4=(88, 58, 4), axes5=(118, 88, 5)
    ),
    # 索引4
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(19, 9, 1), axes2=(39, 19, 2), axes3=(59, 29, 3), axes4=(89, 59, 4), axes5=(119, 89, 5)
    ),
    # 索引5
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(20, 10, 1), axes2=(40, 20, 2), axes3=(60, 30, 3), axes4=(90, 60, 4), axes5=(120, 90, 5)
    ),
    # 索引6
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(21, 11, 1), axes2=(41, 21, 2), axes3=(61, 31, 3), axes4=(91, 61, 4), axes5=(121, 91, 5)
    ),
    # 索引7
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(22, 12, 1), axes2=(42, 22, 2), axes3=(62, 32, 3), axes4=(92, 62, 4), axes5=(122, 92, 5)
    ),
    # 索引8
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(23, 13, 1), axes2=(43, 23, 2), axes3=(63, 33, 3), axes4=(93, 63, 4), axes5=(123, 93, 5)
    ),
    # 索引9
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(24, 14, 1), axes2=(44, 24, 2), axes3=(64, 34, 3), axes4=(94, 64, 4), axes5=(124, 94, 5)
    ),
    # 索引10
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(25, 15, 1), axes2=(45, 25, 2), axes3=(65, 35, 3), axes4=(95, 65, 4), axes5=(125, 95, 5)
    ),
    # 索引11
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(26, 16, 1), axes2=(46, 26, 2), axes3=(66, 36, 3), axes4=(96, 66, 4), axes5=(126, 96, 5)
    ),
    # 索引12
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(27, 17, 1), axes2=(47, 27, 2), axes3=(67, 37, 3), axes4=(97, 67, 4), axes5=(127, 97, 5)
    ),
    # 索引13
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(28, 18, 1), axes2=(48, 28, 2), axes3=(68, 38, 3), axes4=(98, 68, 4), axes5=(128, 98, 5)
    ),
    # 索引14
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(29, 19, 1), axes2=(49, 29, 2), axes3=(69, 39, 3), axes4=(99, 69, 4), axes5=(129, 99, 5)
    ),
    # 索引15（中心组）
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(30, 20, 1), axes2=(50, 30, 2), axes3=(70, 40, 3), axes4=(100, 70, 4), axes5=(130, 100, 5)
    ),
    # 索引16
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(31, 21, 1), axes2=(51, 31, 2), axes3=(71, 41, 3), axes4=(101, 71, 4), axes5=(131, 101, 5)
    ),
    # 索引17
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(32, 22, 1), axes2=(52, 32, 2), axes3=(72, 42, 3), axes4=(102, 72, 4), axes5=(132, 102, 5)
    ),
    # 索引18
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(33, 23, 1), axes2=(53, 33, 2), axes3=(73, 43, 3), axes4=(103, 73, 4), axes5=(133, 103, 5)
    ),
    # 索引19
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(34, 24, 1), axes2=(54, 34, 2), axes3=(74, 44, 3), axes4=(104, 74, 4), axes5=(134, 104, 5)
    ),
    # 索引20
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(35, 25, 1), axes2=(55, 35, 2), axes3=(75, 45, 3), axes4=(105, 75, 4), axes5=(135, 105, 5)
    ),
    # 索引21
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(36, 26, 1), axes2=(56, 36, 2), axes3=(76, 46, 3), axes4=(106, 76, 4), axes5=(136, 106, 5)
    ),
    # 索引22
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(37, 27, 1), axes2=(57, 37, 2), axes3=(77, 47, 3), axes4=(107, 77, 4), axes5=(137, 107, 5)
    ),
    # 索引23
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(38, 28, 1), axes2=(58, 38, 2), axes3=(78, 48, 3), axes4=(108, 78, 4), axes5=(138, 108, 5)
    ),
    # 索引24
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(39, 29, 1), axes2=(59, 39, 2), axes3=(79, 49, 3), axes4=(109, 79, 4), axes5=(139, 109, 5)
    ),
    # 索引25
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(40, 30, 1), axes2=(60, 40, 2), axes3=(80, 50, 3), axes4=(110, 80, 4), axes5=(140, 110, 5)
    ),
    # 索引26
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(41, 31, 1), axes2=(61, 41, 2), axes3=(81, 51, 3), axes4=(111, 81, 4), axes5=(141, 111, 5)
    ),
    # 索引27
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(42, 32, 1), axes2=(62, 42, 2), axes3=(82, 52, 3), axes4=(112, 82, 4), axes5=(142, 112, 5)
    ),
    # 索引28
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(43, 33, 1), axes2=(63, 43, 2), axes3=(83, 53, 3), axes4=(113, 83, 4), axes5=(143, 113, 5)
    ),
    # 索引29
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(44, 34, 1), axes2=(64, 44, 2), axes3=(84, 54, 3), axes4=(114, 84, 4), axes5=(144, 114, 5)
    ),
    # 索引30
    EllipseSet(
        center1=(460, 240), center2=(460, 240), center3=(460, 240), center4=(460, 240), center5=(460, 240),
        axes1=(45, 35, 1), axes2=(65, 45, 2), axes3=(85, 55, 3), axes4=(115, 85, 4), axes5=(145, 115, 5)
    )
]

# 访问方式：
# 第i组第j个椭圆的中心坐标: ellipse_sets[i].center{j}
# 第i组第j个椭圆的轴信息: ellipse_sets[i].axes{j}
# 第i组第j个椭圆的长轴: ellipse_sets[i].axes{j}[0]
# 第i组第j个椭圆的短轴: ellipse_sets[i].axes{j}[1]
# 第i组第j个椭圆的自定义参数: ellipse_sets[i].axes{j}[2]
