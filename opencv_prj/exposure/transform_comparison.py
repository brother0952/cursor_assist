import pygame
import numpy as np
import cv2

class TransformComparison:
    def __init__(self, image_path):
        pygame.init()
        
        # 加载原始图片
        self.original_image = cv2.imread(image_path)
        self.original_height, self.original_width = self.original_image.shape[:2]
        
        # 设置显示缩放比例（如果图片太大）
        self.scale = min(600 / self.original_width, 400 / self.original_height)
        
        # 计算单个显示区域的尺寸
        self.display_width = int(self.original_width * self.scale)
        self.display_height = int(self.original_height * self.scale)
        
        # 创建窗口（左右两个显示区域）
        self.screen = pygame.display.set_mode((self.display_width * 2, self.display_height))
        pygame.display.set_caption("Transform Comparison: Perspective vs Affine")
        
        # 初始化控制点
        self.control_points = {
            'tl': [0, 0],  # 左上
            'tr': [self.original_width-1, 0],  # 右上
            'bl': [0, self.original_height-1],  # 左下
            'br': [self.original_width-1, self.original_height-1]  # 右下
        }
        
        # 拖动状态
        self.dragging = None
        self.control_point_radius = 8
        
        # 颜色定义
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.WHITE = (255, 255, 255)

    def scale_point(self, point, offset_x=0):
        """将原始坐标转换为显示坐标"""
        return (int(point[0] * self.scale) + offset_x, int(point[1] * self.scale))
    
    def unscale_point(self, point, offset_x=0):
        """将显示坐标转换为原始坐标"""
        return ((point[0] - offset_x) / self.scale, point[1] / self.scale)

    def apply_perspective(self):
        """应用透视变换"""
        src_points = np.float32([
            [0, 0],
            [self.original_width-1, 0],
            [0, self.original_height-1],
            [self.original_width-1, self.original_height-1]
        ])
        
        dst_points = np.float32([
            self.control_points['tl'],
            self.control_points['tr'],
            self.control_points['bl'],
            self.control_points['br']
        ])
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(self.original_image, matrix, 
                                   (self.original_width, self.original_height))
        
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        warped = pygame.surfarray.make_surface(warped)
        warped = pygame.transform.scale(warped, (self.display_width, self.display_height))
        
        return warped

    def apply_affine(self):
        """应用仿射变换"""
        src_points = np.float32([
            [0, 0],
            [self.original_width-1, 0],
            [0, self.original_height-1]
        ])
        
        dst_points = np.float32([
            self.control_points['tl'],
            self.control_points['tr'],
            self.control_points['bl']
        ])
        
        matrix = cv2.getAffineTransform(src_points, dst_points)
        warped = cv2.warpAffine(self.original_image, matrix, 
                               (self.original_width, self.original_height))
        
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        warped = pygame.surfarray.make_surface(warped)
        warped = pygame.transform.scale(warped, (self.display_width, self.display_height))
        
        return warped

    def draw_control_points(self, offset_x=0):
        """绘制控制点和连线"""
        points = []
        for name in ['tl', 'tr', 'br', 'bl']:
            scaled_point = self.scale_point(self.control_points[name], offset_x)
            points.append(scaled_point)
        
        pygame.draw.lines(self.screen, self.YELLOW, True, points, 2)
        
        for name, point in self.control_points.items():
            scaled_point = self.scale_point(point, offset_x)
            color = self.RED if self.dragging == name else self.BLUE
            pygame.draw.circle(self.screen, color, scaled_point, self.control_point_radius)
            
            font = pygame.font.Font(None, 24)
            text = font.render(name.upper(), True, color)
            self.screen.blit(text, (scaled_point[0] - 10, scaled_point[1] - 20))

    def get_control_point_at_pos(self, pos):
        """检查位置是否在控制点上（考虑两个显示区域）"""
        for name, point in self.control_points.items():
            # 检查左侧显示区域
            scaled_point = self.scale_point(point)
            distance = np.sqrt((pos[0] - scaled_point[0])**2 + (pos[1] - scaled_point[1])**2)
            if distance < self.control_point_radius:
                return name
                
            # 检查右侧显示区域
            scaled_point = self.scale_point(point, self.display_width)
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
                    if event.button == 1:
                        self.dragging = self.get_control_point_at_pos(event.pos)
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging = None
                
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging:
                        mouse_x = event.pos[0]
                        if mouse_x > self.display_width:
                            new_pos = self.unscale_point(event.pos, self.display_width)
                        else:
                            new_pos = self.unscale_point(event.pos)
                        self.control_points[self.dragging] = list(new_pos)
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # 重置
                        self.control_points = {
                            'tl': [0, 0],
                            'tr': [self.original_width-1, 0],
                            'bl': [0, self.original_height-1],
                            'br': [self.original_width-1, self.original_height-1]
                        }
            
            # 绘制
            self.screen.fill((0, 0, 0))
            
            # 左侧：透视变换
            perspective = self.apply_perspective()
            self.screen.blit(perspective, (0, 0))
            self.draw_control_points()
            
            # 右侧：仿射变换
            affine = self.apply_affine()
            self.screen.blit(affine, (self.display_width, 0))
            self.draw_control_points(self.display_width)
            
            # 显示标题
            font = pygame.font.Font(None, 36)
            text1 = font.render("Perspective", True, self.WHITE)
            text2 = font.render("Affine", True, self.WHITE)
            self.screen.blit(text1, (10, 10))
            self.screen.blit(text2, (self.display_width + 10, 10))
            
            pygame.display.flip()

        pygame.quit()

if __name__ == "__main__":
    editor = TransformComparison("20.jpg")
    editor.run() 