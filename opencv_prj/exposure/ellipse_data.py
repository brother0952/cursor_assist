import numpy as np
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

def init_ellipse_sets(left_set: EllipseSet, center_set: EllipseSet, right_set: EllipseSet) -> List[EllipseSet]:
    """
    根据-15,0,15三个位置的EllipseSet，插值生成所有31个EllipseSet
    """
    sets = []
    
    # 生成15个点的插值（左半部分）
    for i in range(15):
        # 为每个椭圆生成插值后的参数
        axes_list = []
        centers_list = []
        for j in range(1, 6):  # 5个椭圆
            left_axes = getattr(left_set, f'axes{j}')
            center_axes = getattr(center_set, f'axes{j}')
            left_center = getattr(left_set, f'center{j}')
            center_center = getattr(center_set, f'center{j}')
            
            # 对长轴和短轴进行插值
            major_axis = int(np.interp(i, [0, 14], [left_axes[0], center_axes[0]]))
            minor_axis = int(np.interp(i, [0, 14], [left_axes[1], center_axes[1]]))
            
            # 对center坐标进行插值
            center_x = int(np.interp(i, [0, 14], [left_center[0], center_center[0]]))
            center_y = int(np.interp(i, [0, 14], [left_center[1], center_center[1]]))
            
            axes_list.append((major_axis, minor_axis, j))  # j作为自定义参数
            centers_list.append((center_x, center_y))
            
        # 创建新的EllipseSet实例
        new_set = EllipseSet(
            center1=centers_list[0], center2=centers_list[1], 
            center3=centers_list[2], center4=centers_list[3], 
            center5=centers_list[4],
            axes1=axes_list[0], axes2=axes_list[1], axes3=axes_list[2], 
            axes4=axes_list[3], axes5=axes_list[4]
        )
        sets.append(new_set)
    
    # 添加中心点
    sets.append(center_set)
    
    # 生成15个点的插值（右半部分）
    for i in range(15):
        axes_list = []
        centers_list = []
        for j in range(1, 6):
            center_axes = getattr(center_set, f'axes{j}')
            right_axes = getattr(right_set, f'axes{j}')
            center_center = getattr(center_set, f'center{j}')
            right_center = getattr(right_set, f'center{j}')
            
            # 对长轴和短轴进行插值
            major_axis = int(np.interp(i, [0, 14], [center_axes[0], right_axes[0]]))
            minor_axis = int(np.interp(i, [0, 14], [center_axes[1], right_axes[1]]))
            
            # 对center坐标进行插值
            center_x = int(np.interp(i, [0, 14], [center_center[0], right_center[0]]))
            center_y = int(np.interp(i, [0, 14], [center_center[1], right_center[1]]))
            
            axes_list.append((major_axis, minor_axis, j))
            centers_list.append((center_x, center_y))
            
        new_set = EllipseSet(
            center1=centers_list[0], center2=centers_list[1], 
            center3=centers_list[2], center4=centers_list[3], 
            center5=centers_list[4],
            axes1=axes_list[0], axes2=axes_list[1], axes3=axes_list[2], 
            axes4=axes_list[3], axes5=axes_list[4]
        )
        sets.append(new_set)
    
    return sets

def print_ellipse_sets(sets: List[EllipseSet]):
    """
    打印所有EllipseSet的值，采用紧凑格式
    """
    print("\nEllipse Sets Values:")
    print("-" * 120)
    
    # 打印表头
    print(f"{'Index':>6} {'Position':>8} | {'Centers':^55} | {'Axes (major, minor, param)':^50}")
    print("-" * 120)
    
    for i, set_data in enumerate(sets):
        index = i - 15  # 将0-30的索引转换为-15到+15
        
        # 格式化centers字符串
        centers = f"c1{set_data.center1}, c2{set_data.center2}, c3{set_data.center3}, c4{set_data.center4}, c5{set_data.center5}"
        
        # 格式化axes字符串
        axes = f"a1{set_data.axes1}, a2{set_data.axes2}, a3{set_data.axes3}, a4{set_data.axes4}, a5{set_data.axes5}"
        
        # 分两行显示，第一行显示centers，第二行显示axes
        print(f"{i:>6} {index:>8} | {centers:<55} |")
        print(f"{'':>16} | {axes:<105} |")
        print("-" * 120)

# 示例使用：
left_set = EllipseSet(
    center1=(-50, 0), center2=(0, 0), center3=(0, 0), center4=(0, 0), center5=(0, 0),
    axes1=(15, 5, 1), axes2=(35, 15, 2), axes3=(55, 25, 3), axes4=(85, 55, 4), axes5=(115, 85, 5)
)

center_set = EllipseSet(
    center1=(0, 0), center2=(0, 0), center3=(0, 0), center4=(0, 0), center5=(0, 0),
    axes1=(25, 12, 1), axes2=(60, 16, 2), axes3=(90, 20, 3), axes4=(120, 28, 4), axes5=(150, 36, 5)
)

right_set = EllipseSet(
    center1=(50, 0), center2=(0, 0), center3=(0, 0), center4=(0, 0), center5=(0, 0),
    axes1=(45, 35, 1), axes2=(65, 45, 2), axes3=(85, 55, 3), axes4=(115, 85, 4), axes5=(145, 115, 5)
)

# 生成所有31个EllipseSet
ellipse_sets = init_ellipse_sets(left_set, center_set, right_set)
print_ellipse_sets(ellipse_sets)

# 访问方式：
# 第i组第j个椭圆的中心坐标: ellipse_sets[i].center{j}
# 第i组第j个椭圆的轴信息: ellipse_sets[i].axes{j}
# 第i组第j个椭圆的长轴: ellipse_sets[i].axes{j}[0]
# 第i组第j个椭圆的短轴: ellipse_sets[i].axes{j}[1]
# 第i组第j个椭圆的自定义参数: ellipse_sets[i].axes{j}[2]
