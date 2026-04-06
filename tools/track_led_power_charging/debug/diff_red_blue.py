import cv2
import numpy as np


def filter_rgb_brightness(frame):
    # lower=np.array([240,240,240])
    # upper=np.array([255,255,255])
    # mask_brightness = cv2.inRange(frame, lower, upper)
    # return mask_brightness
    return frame

def detect_led_color(frame):
    # 读取图像
    # image = cv2.imread(image_path)

    rgb_brightness=filter_rgb_brightness(frame)
    
    # 转换为HSV颜色空间
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 生成亮度Mask（假设LED亮度较高，V值>200）
    brightness_lower = 255  # 根据实际LED亮度调整
    brightness_mask = cv2.inRange(hsv_image[:,:,2], brightness_lower, 255)

    # 可选：形态学操作优化亮度Mask（去除小噪点）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    brightness_mask = cv2.morphologyEx(brightness_mask, cv2.MORPH_CLOSE, kernel)

    
    
    # 定义红色和蓝色的HSV范围
    lower_red1 = np.array([0, 0, 255])
    upper_red1 = np.array([10, 120, 255])
    lower_red2 = np.array([170, 0, 255])
    upper_red2 = np.array([180, 120, 255])
    
    lower_blue = np.array([100, 0, 255])
    upper_blue = np.array([120, 120, 255])
    
    # 创建颜色掩膜
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2) # bitwise_or
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
    

    final_red_mask = cv2.bitwise_and(mask_red, brightness_mask)
    final_blue_mask = cv2.bitwise_and(mask_blue, brightness_mask)

    # 计算像素数量
    red_pixels = cv2.countNonZero(final_red_mask)
    blue_pixels = cv2.countNonZero(final_blue_mask)
    
    # 判断LED颜色
    if red_pixels > blue_pixels and red_pixels > 0:
        color = "red"
    elif blue_pixels > red_pixels and blue_pixels > 0:
        color = "blue"
    else:
        color = "未识别"

    # brightness_mask=cv2.bitwise_or(brightness_mask,rgb_brightness)
    
    # 显示结果
    cv2.imshow("brightness_mask mask", brightness_mask)
    cv2.imshow("Red Mask", final_red_mask)
    cv2.imshow("Blue Mask", final_blue_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return color

import cv2
import numpy as np

# Global variable to store the current frame
current_frame = None

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function to print RGB values when clicking on the frame
    """
    global current_frame
    
    if event == cv2.EVENT_LBUTTONDOWN and current_frame is not None:
        # Get BGR values (OpenCV uses BGR by default)
        b, g, r = current_frame[y, x]
        print(f"Position: ({x}, {y}) - RGB Values: R={r}, G={g}, B={b}")

def main():
    """
    Main function to capture video and print RGB values on click
    """
    global current_frame
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Debug RGB Values')
    cv2.setMouseCallback('Debug RGB Values', mouse_callback)
    
    print("Left click on the image to print RGB values at that position")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Store the current frame for the mouse callback
        current_frame = frame.copy()
        res=detect_led_color(current_frame)
        # print(res)
        
        # Display the frame
        cv2.imshow('Debug RGB Values', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()