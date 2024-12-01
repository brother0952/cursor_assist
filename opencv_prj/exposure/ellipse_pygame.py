import pygame
import numpy as np
import cv2

class EllipseEditor:
    def __init__(self):
        pygame.init()
        
        # 设置原始画布大小和缩放比例
        self.original_width = 320
        self.original_height = 80
        self.scale = 1
        
        # 计算实际显示大小
        self.width = self.original_width * self.scale
        self.height = self.original_height * self.scale
        
        # 创建窗口
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Ellipse Perspective Editor")
        
        # 椭圆参数
        self.center = [self.original_width // 2, self.original_height // 2]
        self.axes = [20, 12]  # [major_axis, minor_axis]
        
        # 计算椭圆的边界框
        self.update_bounding_rect()
        
        # 透视变换参数（基于边界框）
        self.perspective_points = self.get_initial_perspective_points()
        
        # 拖动状态
        self.dragging = None
        self.control_point_radius = 5
        
        # 颜色定义
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.WHITE = (255, 255, 255)
        self.YELLOW = (255, 255, 0)

    def update_bounding_rect(self):
        """更新椭圆的边界框，确保完全包围椭圆"""
        # 添加一些边距确保完全包围
        margin = 0
        self.bbox_left = self.center[0] - self.axes[0] - margin
        self.bbox_right = self.center[0] + self.axes[0] + margin
        self.bbox_top = self.center[1] - self.axes[1] - margin
        self.bbox_bottom = self.center[1] + self.axes[1] + margin

        print("x diretion",self.bbox_left,self.center[0],self.axes[0])

    def get_initial_perspective_points(self):
        """获取基于边界框的初始透视点"""
        return {
            'tl': [self.bbox_left, self.bbox_top],
            'tr': [self.bbox_right, self.bbox_top],
            'bl': [self.bbox_left, self.bbox_bottom],
            'br': [self.bbox_right, self.bbox_bottom]
        }

    def draw_transformed_ellipse(self):
        """绘制透视变换后的椭圆"""
        # 创建原始椭圆，使用边界框大小而不是整个画布
        bbox_width = int(self.bbox_right - self.bbox_left + 1)
        bbox_height = int(self.bbox_bottom - self.bbox_top + 1)
        
        # 创建刚好包含椭圆的画布
        canvas = np.zeros((bbox_height, bbox_width, 3), dtype=np.uint8)
        
        # 调整椭圆中心点到边界框坐标系
        relative_center = (
            int(self.center[0] - self.bbox_left),
            int(self.center[1] - self.bbox_top)
        )
        
        # 绘制椭圆
        cv2.ellipse(canvas, 
                    relative_center,
                    (int(self.axes[0]), int(self.axes[1])),
                    0, 0, 360, (0, 255, 0), -1)
        
        # 应用透视变换
        src_points = np.float32([
            [0, 0],
            [bbox_width-1, 0],
            [0, bbox_height-1],
            [bbox_width-1, bbox_height-1]
        ])
        
        dst_points = np.float32([
            self.perspective_points['tl'],
            self.perspective_points['tr'],
            self.perspective_points['bl'],
            self.perspective_points['br']
        ])
        
        # matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        # transformed = cv2.warpPerspective(canvas, matrix, 
        #                                 (self.original_width, self.original_height))
        
        # 转换为Pygame surface
        transformed = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        surface = pygame.surfarray.make_surface(transformed)
        surface = pygame.transform.scale(surface, (self.width, self.height))
        
        return surface

    def draw_bounding_rect(self):
        """绘制原始边界框"""
        points = [
            (self.bbox_left, self.bbox_top),
            (self.bbox_right, self.bbox_top),
            (self.bbox_right, self.bbox_bottom),
            (self.bbox_left, self.bbox_bottom)
        ]
        scaled_points = [self.scale_point(p) for p in points]
        pygame.draw.lines(self.screen, (128, 128, 128), True, scaled_points, 1)

    def scale_point(self, point):
        return (int(point[0] * self.scale), int(point[1] * self.scale))
    
    def unscale_point(self, point):
        return (point[0] / self.scale, point[1] / self.scale)

    def draw_control_points(self):
        """绘制控制点和变形框"""
        # 绘制变形框
        points = []
        for name in ['tl', 'tr', 'br', 'bl']:
            scaled_point = self.scale_point(self.perspective_points[name])
            points.append(scaled_point)
        
        # 绘制框线
        pygame.draw.lines(self.screen, self.YELLOW, True, points, 1)
        
        # 绘制控制点
        for name, point in self.perspective_points.items():
            scaled_point = self.scale_point(point)
            color = self.RED if self.dragging == name else self.BLUE
            pygame.draw.circle(self.screen, color, scaled_point, self.control_point_radius)
            
            # 绘制控制点标签
            font = pygame.font.Font(None, 24)
            text = font.render(name.upper(), True, color)
            self.screen.blit(text, (scaled_point[0] - 10, scaled_point[1] - 20))

    def get_control_point_at_pos(self, pos):
        """检查位置是否在控制点上"""
        for name, point in self.perspective_points.items():
            scaled_point = self.scale_point(point)
            distance = np.sqrt((pos[0] - scaled_point[0])**2 + (pos[1] - scaled_point[1])**2)
            if distance < self.control_point_radius:
                return name
        return None

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # 左键
                        self.dragging = self.get_control_point_at_pos(event.pos)
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # 左键
                        self.dragging = None
                
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging:
                        # 更新控制点位置
                        new_pos = self.unscale_point(event.pos)
                        self.perspective_points[self.dragging] = list(new_pos)
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # 重置
                        self.perspective_points = self.get_initial_perspective_points()
                    elif event.key == pygame.K_s:  # 保存参数
                        print("Perspective points:", self.perspective_points)
            
            # 绘制
            self.screen.fill(self.BLACK)
            transformed = self.draw_transformed_ellipse()
            self.screen.blit(transformed, (0, 0))
            self.draw_bounding_rect()  # 显示原始边界框
            self.draw_control_points()
            
            # 显示当前控制点位置
            if self.dragging:
                font = pygame.font.Font(None, 24)
                pos = self.perspective_points[self.dragging]
                text = font.render(f"{self.dragging.upper()}: ({int(pos[0])}, {int(pos[1])})", True, self.WHITE)
                self.screen.blit(text, (10, 10))
            
            pygame.display.flip()

        pygame.quit()

if __name__ == "__main__":
    editor = EllipseEditor()
    editor.run() 