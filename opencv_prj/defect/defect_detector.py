import cv2
import numpy as np

class DefectDetector:
    def __init__(self):
        self.min_area = 10
        self.tolerance = 1.5  # 距离容差系数
        
    def find_points(self, binary):
        """查找所有点的坐标"""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 获取所有点的中心坐标
        points = []
        for cnt in contours:
            if cv2.contourArea(cnt) > self.min_area:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    points.append((cx, cy))
        
        return np.array(points)
    
    def group_points_by_rows(self, points, tolerance=5):
        """将点按行分组"""
        # 按y坐标排序
        sorted_points = points[np.argsort(points[:, 1])]
        
        rows = []
        current_row = [sorted_points[0]]
        current_y = sorted_points[0][1]
        
        # 遍历所有点，按y坐标分组
        for point in sorted_points[1:]:
            if abs(point[1] - current_y) <= tolerance:
                current_row.append(point)
            else:
                # 将当前行按x坐标排序
                rows.append(sorted(current_row, key=lambda p: p[0]))
                current_row = [point]
                current_y = point[1]
        
        # 添加最后一行
        if current_row:
            rows.append(sorted(current_row, key=lambda p: p[0]))
        
        return rows
    
    def find_missing_points(self, rows):
        """检测每行中缺失的点"""
        defects = []
        
        for row in rows:
            if len(row) < 3:  # 忽略太短的行
                continue
            
            # 计算该行中点之间的平均间距
            distances = []
            for i in range(len(row)-1):
                dist = row[i+1][0] - row[i][0]
                distances.append(dist)
            
            avg_distance = np.median(distances)  # 使用中位数作为标准间距
            
            # 检查每对相邻点之间的距离
            for i in range(len(row)-1):
                current_dist = row[i+1][0] - row[i][0]
                
                # 如果距离明显大于平均值，说明中间缺少点
                if current_dist > avg_distance * self.tolerance:
                    # 计算缺失点的数量
                    missing_count = round(current_dist / avg_distance) - 1
                    
                    # 计算缺失点的位置
                    for j in range(missing_count):
                        x = row[i][0] + avg_distance * (j + 1)
                        y = (row[i][1] + row[i+1][1]) / 2
                        defects.append((int(x), int(y)))
        
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
        rows = self.group_points_by_rows(points)
        
        # 检测缺失的点
        defects = self.find_missing_points(rows)
        
        # 在原图上标记结果
        result = image.copy()
        
        # 标记所有检测到的点
        for point in points:
            cv2.circle(result, tuple(point), 2, (0, 255, 0), -1)
        
        # 标记缺陷
        for x, y in defects:
            cv2.circle(result, (x, y), 8, (0, 0, 255), 2)
            cv2.line(result, (x-6, y), (x+6, y), (0, 0, 255), 2)
            cv2.line(result, (x, y-6), (x, y+6), (0, 0, 255), 2)
        
        # 创建可视化掩码
        mask = np.zeros_like(gray)
        
        # 在掩码上标记点和缺陷
        for point in points:
            cv2.circle(mask, tuple(point), 3, 255, -1)
        for x, y in defects:
            cv2.circle(mask, (x, y), 5, 200, 2)
        
        # 在掩码上绘制行
        for row in rows:
            if len(row) >= 2:
                for i in range(len(row)-1):
                    pt1 = tuple(row[i])
                    pt2 = tuple(row[i+1])
                    cv2.line(mask, pt1, pt2, 64, 1)
        
        return result, mask, defects

def main():
    detector = DefectDetector()
    
    try:
        image_path = input("请输入图像路径: ")
        result, mask, defects = detector.detect(image_path)
        
        # 显示结果
        cv2.imshow("原图及缺陷标记", result)
        cv2.imshow("点阵分析", mask)
        
        # 保存结果
        output_path = "defect_result.png"
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