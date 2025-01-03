import cv2
import numpy as np

class ColumnDefectDetector:
    def __init__(self):
        self.min_area = 10
        self.params = {
            'row_tolerance': 10,      # y方向容差
            'x_tolerance': 32,       # x方向容差
            'gap_threshold': 26,     # 基准间距
            'edge_margin': 30,       # 边缘区域大小
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
    
    def group_points_by_row(self, points):
        """将点按行分组，考虑x和y坐标"""
        # 先按y坐标粗分组
        sorted_points = points[np.argsort(points[:, 1])]
        
        # 临时存储粗分组
        temp_rows = []
        current_row = []
        current_y = None
        tolerance = self.params['row_tolerance']
        
        # 第一次按y坐标粗分组
        for point in sorted_points:
            x, y = point
            
            if current_y is None:
                current_y = y
                current_row = [point]
            elif abs(y - current_y) <= tolerance:
                current_row.append(point)
            else:
                if len(current_row) >= 2:
                    temp_rows.append(current_row)
                current_row = [point]
                current_y = y
        
        if current_row and len(current_row) >= 2:
            temp_rows.append(current_row)
        
        # 第二次处理：对每个粗分组进行细分
        final_rows = []
        for temp_row in temp_rows:
            # 按x坐标排序
            sorted_by_x = sorted(temp_row, key=lambda p: p[0])
            
            # 检查x方向的连续性
            sub_rows = []
            current_sub_row = [sorted_by_x[0]]
            current_x = sorted_by_x[0][0]
            x_tolerance = self.params.get('x_tolerance', 30)  # x方向的容差
            
            for point in sorted_by_x[1:]:
                x, y = point
                if abs(x - current_x - self.params['gap_threshold']) <= x_tolerance:
                    current_sub_row.append(point)
                    current_x = x
                else:
                    if len(current_sub_row) >= 2:
                        sub_rows.append(current_sub_row)
                    current_sub_row = [point]
                    current_x = x
            
            if current_sub_row and len(current_sub_row) >= 2:
                sub_rows.append(current_sub_row)
            
            # 将有效的子行添加到最终结果
            final_rows.extend(sub_rows)
        
        # 打印分组信息
        print(f"找到 {len(final_rows)} 行")
        for i, row in enumerate(final_rows):
            print(f"行 {i}: {len(row)} 个点, y={row[0][1]}")
        
        return final_rows
    
    def analyze_horizontal_gaps(self, row):
        """分析一行中的水平间距，检测缺陷"""
        defects = []
        if len(row) < 3:  # 至少需要3个点才能判断
            return defects
        
        # 计算所有间距
        gaps = []
        for i in range(len(row)-1):
            gap = row[i+1][0] - row[i][0]
            gaps.append(gap)
        
        # 使用众数作为标准间距
        hist, bins = np.histogram(gaps, bins=20)
        standard_gap = bins[np.argmax(hist)]
        
        print(f"标准间距: {standard_gap:.2f}")
        
        # 检查相邻点之间的间距
        for i in range(len(row)-1):
            current_gap = row[i+1][0] - row[i][0]
            if current_gap > standard_gap * 1.5:  # 如果间距过大
                # 计算应该有多少个点
                missing_count = round(current_gap / standard_gap) - 1
                print(f"发现间距: {current_gap:.2f}, 标准: {standard_gap:.2f}, 缺失: {missing_count}")
                
                # 添加缺失的点
                for j in range(missing_count):
                    x = int(row[i][0] + standard_gap * (j + 1))
                    y = int((row[i][1] + row[i+1][1]) / 2)
                    defects.append((x, y))
        
        return defects
    
    def detect(self, image_path):
        """检测图像中的缺陷"""
        # 读取并预处理图像
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("无法读取图像")
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        
        # 查找所有点
        points = self.find_points(binary)
        if len(points) == 0:
            raise Exception("未找到任何点")
        
        # 按行分组
        rows = self.group_points_by_row(points)
        
        # 在原图上标记结果
        result = image.copy()
        mask = np.zeros_like(gray)
        
        # 用不同颜色标记不同行的点并连线
        colors = [(0,0,255), (0,255,0), (255,0,0), (255,255,0), 
                 (255,0,255), (0,255,255), (128,128,0), (0,128,128)]
        
        # 分析每行并收集缺陷
        all_defects = []
        for i, row in enumerate(rows):
            color = colors[i % len(colors)]
            
            # 画出这一行的点和连线
            for j in range(len(row)):
                cv2.circle(result, tuple(row[j]), 2, color, -1)
                if j < len(row) - 1:
                    cv2.line(result, tuple(row[j]), tuple(row[j+1]), color, 1)
            
            # 检测缺陷
            defects = self.analyze_horizontal_gaps(row)
            all_defects.extend(defects)
        
        # 标记缺陷
        for x, y in all_defects:
            cv2.circle(result, (x, y), 8, (0, 0, 255), 2)
            cv2.line(result, (x-6, y), (x+6, y), (0, 0, 255), 2)
            cv2.line(result, (x, y-6), (x, y+6), (0, 0, 255), 2)
            cv2.circle(mask, (x, y), 5, 200, 2)
        
        # 显示行分组结果
        cv2.imshow("行分组结果", result)
        
        return result, mask, all_defects

def main():
    detector = ColumnDefectDetector()
    
    try:
        # image_path = input("请输入图像路径: ")
        image_path = r"D:\git\py_process\opencv_prj\defect\kd3f6nh06a.png"
        result, mask, defects = detector.detect(image_path)
        
        # 显示结果
        cv2.imshow("原图及缺陷标记", result)
        cv2.imshow("点阵分析", mask)
        
        # 保存结果
        output_path = "column_defect_result.png"
        cv2.imwrite(output_path, result)
        print(f"检测到 {len(defects)} 个缺陷")
        print(f"缺陷位置: {defects}")
        print(f"结果已保存至: {output_path}")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main() 