import cv2
import datetime
import os

def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    # 获取默认分辨率
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    print(f"摄像头默认分辨率: {int(width)} x {int(height)}")
    
    print("按 'S' 键保存当前画面，按 'Q' 键退出")
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        
        if not ret:
            print("错误：无法读取画面")
            break
        
        # 显示实时画面
        cv2.imshow('Camera Debugger - Press S to save, Q to quit', frame)
        
        # 等待按键输入
        key = cv2.waitKey(1) & 0xFF
        
        # 按 'Q' 键退出
        if key == ord('q') or key == ord('Q'):
            break
            
        # 按 'S' 键保存画面
        elif key == ord('s') or key == ord('S'):
            # 使用当前日期时间命名文件
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_capture_{timestamp}.jpg"
            
            # 保存图片
            success = cv2.imwrite(filename, frame)
            
            if success:
                print(f"画面已保存为: {filename}")
            else:
                print("保存失败")
    
    # 释放摄像头资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()