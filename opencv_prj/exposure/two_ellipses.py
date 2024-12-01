import cv2
import numpy as np

def special_ellipse(canvas, center:tuple, axesx_left, axesx_right,axesy, color):
    cv2.ellipse(canvas, 
                center, 
                (axesx_right, axesy), 
                180,  # 角度
                90,  # 起始角
                270,  # 结束角
                color, 
                -1)  # 填充
    
    cv2.ellipse(canvas, 
                center, 
                (axesx_left, axesy), 
                0, 
                90, 
                270, 
                color, 
                -1)
    pass

def main():
    # 创建画布
    canvas_width = 1000
    canvas_height = 800
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # 设置椭圆参数
    ellipse1 = {
        'center': (canvas_width // 3, canvas_height // 2),  # 在画布左侧1/3处
        'axes': (100, 80),  # (长轴, 短轴)
        'color': (0, 255, 0)  # 绿色
    }
    
    ellipse2 = {
        # 'center': (2 * canvas_width // 3, canvas_height // 2),  # 在画布右侧1/3处
        'center': (300,100),
        'axes': (300, 80),  # (长轴, 短轴)
        'color': (0, 255, 255)  # yellow
    }
    
    # 绘制椭圆
    # cv2.ellipse(canvas, 
    #             ellipse1['center'], 
    #             ellipse1['axes'], 
    #             0,  # 角度
    #             90,  # 起始角
    #             270,  # 结束角
    #             ellipse1['color'], 
    #             -1)  # 填充
    
    # cv2.ellipse(canvas, 
    #             ellipse2['center'], 
    #             ellipse2['axes'], 
    #             0, 
    #             90, 
    #             270, 
    #             ellipse2['color'], 
    #             -1)

    special_ellipse(canvas, ellipse1['center'], 200, 300, 80, (0, 0, 255))            
          
    
    # 显示参数信息
    info_text = [
        f'Ellipse 1: center={ellipse1["center"]}, axes={ellipse1["axes"]}',
        f'Ellipse 2: center={ellipse2["center"]}, axes={ellipse2["axes"]}'
    ]
    
    for i, text in enumerate(info_text):
        cv2.putText(canvas, text, (10, 30 + i*30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 显示画布
    cv2.imshow('Two Ellipses', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 