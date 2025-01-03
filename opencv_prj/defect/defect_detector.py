import cv2
import numpy as np

class DefectDetector:
    def __init__(self):
        self.min_area = 10
        # 使用字典存储参数，方便通过滑动条更新
        self.params = {
            'h_tolerance': 18,  # 水平容差 (实际值除以10)
            'v_tolerance': 15,  # 垂直容差 (实际值除以10)
            'neighbor_threshold': 6,  # 邻近点验证阈值
            'row_tolerance': 5,  # 行分组容差
            'min_spacing': 5,  # 最小网格间距
            'max_spacing': 50   # 最大网格间距
        }
        
    def find_points(self, binary):
        """查找所有点的坐标"""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        points = []
        for cnt in contours:
            if cv2.contourArea(cnt) > self.min_area:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    points.append((cx, cy))
        
        return np.array(points)
    
    def group_points_by_rows(self, points):
        """将点按行分组，考虑图像倾斜"""
        sorted_points = points[np.argsort(points[:, 1])]
        
        rows = []
        current_row = [sorted_points[0]]
        current_y = sorted_points[0][1]
        
        for point in sorted_points[1:]:
            if abs(point[1] - current_y) <= self.params['row_tolerance']:
                current_row.append(point)
            else:
                if len(current_row) >= 3:  # 至少3个点才能构成有效行
                    rows.append(sorted(current_row, key=lambda p: p[0]))
                current_row = [point]
                current_y = point[1]
        
        if len(current_row) >= 3:
            rows.append(sorted(current_row, key=lambda p: p[0]))
        
        return rows
    
    def get_grid_parameters(self, points):
        """计算网格参数"""
        # 计算水平和垂直方向的间距
        x_diffs = []
        y_diffs = []
        
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dx = abs(points[i][0] - points[j][0])
                dy = abs(points[i][1] - points[j][1])
                if self.params['min_spacing'] < dx < self.params['max_spacing']:  # 合理的水平间距范围
                    x_diffs.append(dx)
                if self.params['min_spacing'] < dy < self.params['max_spacing']:  # 合理的垂直间距范围
                    y_diffs.append(dy)
        
        # 使用直方图找出最常见的间距
        if x_diffs:
            hist_x, bins_x = np.histogram(x_diffs, bins=20)
            x_spacing = bins_x[np.argmax(hist_x)]
        else:
            x_spacing = None
            
        if y_diffs:
            hist_y, bins_y = np.histogram(y_diffs, bins=20)
            y_spacing = bins_y[np.argmax(hist_y)]
        else:
            y_spacing = None
            
        return x_spacing, y_spacing

    def verify_missing_point(self, x, y, points, x_spacing, y_spacing):
        """验证缺失点的有效性"""
        neighbor_count = 0
        
        # 检查8个方向的邻近点
        for px, py in points:
            dx = abs(px - x)
            dy = abs(py - y)
            
            # 检查点是否在合理的网格距离内
            if (x_spacing * 0.8 <= dx <= x_spacing * 1.2 and dy <= y_spacing * 0.3) or \
               (y_spacing * 0.8 <= dy <= y_spacing * 1.2 and dx <= x_spacing * 0.3) or \
               (x_spacing * 0.8 <= dx <= x_spacing * 1.2 and 
                y_spacing * 0.8 <= dy <= y_spacing * 1.2):  # 对角线方向
                neighbor_count += 1
                if neighbor_count >= self.params['neighbor_threshold']:
                    return True
        return False

    def find_missing_points(self, rows, all_points):
        """检测缺失的点"""
        defects = []
        x_spacing, y_spacing = self.get_grid_parameters(all_points)
        
        if x_spacing is None or y_spacing is None:
            return defects
            
        # 检查水平方向的缺失点
        for row in rows:
            if len(row) < 5:
                continue
                
            for i in range(len(row)-1):
                current_dist = row[i+1][0] - row[i][0]
                if current_dist > x_spacing * self.params['h_tolerance'] / 10:
                    missing_count = round(current_dist / x_spacing) - 1
                    for j in range(missing_count):
                        x = int(row[i][0] + x_spacing * (j + 1))
                        y = int((row[i][1] + row[i+1][1]) / 2)
                        if self.verify_missing_point(x, y, all_points, x_spacing, y_spacing):
                            defects.append((x, y))
        
        return defects

    def process_image(self, image):
        """处理图像并返回结果（用于实时更新）"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        
        points = self.find_points(binary)
        if len(points) == 0:
            return image.copy(), np.zeros_like(gray), []
        
        rows = self.group_points_by_rows(points)
        defects = self.find_missing_points(rows, points)
        
        result = image.copy()
        mask = np.zeros_like(gray)
        
        # 标记检测到的点
        for point in points:
            cv2.circle(result, tuple(point), 2, (0, 255, 0), -1)
            cv2.circle(mask, tuple(point), 3, 255, -1)
        
        # 标记缺陷
        for x, y in defects:
            cv2.circle(result, (x, y), 8, (0, 0, 255), 2)
            cv2.line(result, (x-6, y), (x+6, y), (0, 0, 255), 2)
            cv2.line(result, (x, y-6), (x, y+6), (0, 0, 255), 2)
            cv2.circle(mask, (x, y), 5, 200, 2)
            
        return result, mask, defects

def create_parameter_window():
    """创建参数调整窗口"""
    cv2.namedWindow('Parameters')
    
    # 创建滑动条
    cv2.createTrackbar('H_Tolerance(x0.1)', 'Parameters', 18, 30, lambda x: None)
    cv2.createTrackbar('V_Tolerance(x0.1)', 'Parameters', 15, 30, lambda x: None)
    cv2.createTrackbar('Neighbor_Threshold', 'Parameters', 6, 12, lambda x: None)
    cv2.createTrackbar('Row_Tolerance', 'Parameters', 5, 15, lambda x: None)
    cv2.createTrackbar('Min_Spacing', 'Parameters', 5, 20, lambda x: None)
    cv2.createTrackbar('Max_Spacing', 'Parameters', 50, 100, lambda x: None)

def get_parameters():
    """获取滑动条的参数值"""
    params = {}
    # params['h_tolerance'] = cv2.getTrackbarPos('H_Tolerance(x0.1)', 'Parameters')
    # params['v_tolerance'] = cv2.getTrackbarPos('V_Tolerance(x0.1)', 'Parameters')
    # params['neighbor_threshold'] = cv2.getTrackbarPos('Neighbor_Threshold', 'Parameters')
    # params['row_tolerance'] = cv2.getTrackbarPos('Row_Tolerance', 'Parameters')
    # params['min_spacing'] = cv2.getTrackbarPos('Min_Spacing', 'Parameters')
    # params['max_spacing'] = cv2.getTrackbarPos('Max_Spacing', 'Parameters')
    params['h_tolerance'] = 4
    params['v_tolerance'] = 15
    params['neighbor_threshold'] = 7
    params['row_tolerance'] = 10
    params['min_spacing'] = 12
    params['max_spacing'] = 47
    return params

def main():
    detector = DefectDetector()
    
    try:
        # image_path = input("请输入图像路径: ")
        image_path = r"D:\git\py_process\opencv_prj\defect\kd3f6nh06a.png"
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("无法读取图像")
        
        # 创建参数调整窗口
        # create_parameter_window()
        
        while True:
            # 获取当前参数
            detector.params = get_parameters()
            
            # 处理图像
            result, mask, defects = detector.process_image(image)
            
            # 显示结果
            cv2.imshow("原图及缺陷标记", result)
            cv2.imshow("点阵分析", mask)
            
            # 显示检测结果
            info_image = np.zeros((100, 400, 3), dtype=np.uint8)
            cv2.putText(info_image, f"Defects found: {len(defects)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Info", info_image)
            
            key = cv2.waitKey(100) & 0xFF
            if key == 27:  # ESC键退出
                break
            elif key == ord('s'):  # 's'键保存结果
                # cv2.imwrite("defect_result.png", result)
                print(f"检测到 {len(defects)} 个缺陷")
                print(f"缺陷位置: {defects}")
                print(f"当前参数:")
                for k, v in detector.params.items():
                    print(f"  {k}: {v}")
                print("结果已保存至: defect_result.png")
        
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main() 