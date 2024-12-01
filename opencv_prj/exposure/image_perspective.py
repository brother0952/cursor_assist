import pygame
import numpy as np
import cv2

class ImagePerspectiveEditor:
    def __init__(self, image_path):
        pygame.init()
        
        # 加载原始图片
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            print(f"Error: Could not load image '{image_path}'")
            exit(1)
            
        self.original_height, self.original_width = self.original_image.shape[:2]
        
        # 设置显示缩放比例（如果图片太大）
        self.scale = min(1200 / self.original_width, 800 / self.original_height)
        
        # 计算显示尺寸
        self.display_width = int(self.original_width * self.scale)
        self.display_height = int(self.original_height * self.scale)
        
        # 创建窗口
        self.screen = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption("Image Stretch Editor")
        
        # 初始化控制点（图片四个角）
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
        
        # 转换原始图像为RGB并创建pygame surface
        self.image_surface = self.convert_image_to_surface(self.original_image)

    def convert_image_to_surface(self, image):
        """将OpenCV图像转换为Pygame surface"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (self.display_width, self.display_height))
        return pygame.surfarray.make_surface(image_rgb)

    def scale_point(self, point):
        return (int(point[0] * self.scale), int(point[1] * self.scale))
    
    def unscale_point(self, point):
        return (point[0] / self.scale, point[1] / self.scale)

    def draw_stretched_image(self):
        """使用仿射变换绘制变形后的图像"""
        # 获取源点和目标点
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
        
        # 计算仿射变换矩阵
        matrix = cv2.getAffineTransform(src_points, dst_points)
        
        # 应用变换
        warped = cv2.warpAffine(self.original_image, matrix, 
                               (self.original_width, self.original_height))
        
        # 转换为pygame surface
        return self.convert_image_to_surface(warped)

    def draw_control_points(self):
        """绘制控制点和连线"""
        points = []
        for name in ['tl', 'tr', 'br', 'bl']:
            scaled_point = self.scale_point(self.control_points[name])
            points.append(scaled_point)
        
        # 绘制边框
        pygame.draw.lines(self.screen, self.YELLOW, True, points, 2)
        
        # 绘制控制点
        for name, point in self.control_points.items():
            scaled_point = self.scale_point(point)
            color = self.RED if self.dragging == name else self.BLUE
            pygame.draw.circle(self.screen, color, scaled_point, self.control_point_radius)
            
            font = pygame.font.Font(None, 24)
            text = font.render(name.upper(), True, color)
            self.screen.blit(text, (scaled_point[0] - 10, scaled_point[1] - 20))

    def get_control_point_at_pos(self, pos):
        for name, point in self.control_points.items():
            scaled_point = self.scale_point(point)
            distance = np.sqrt((pos[0] - scaled_point[0])**2 + (pos[1] - scaled_point[1])**2)
            if distance < self.control_point_radius:
                return name
        return None

    def run(self):
        clock = pygame.time.Clock()
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
                    elif event.key == pygame.K_s:  # 保存
                        pygame.image.save(self.screen, 'stretched.jpg')
                        print("Saved as stretched.jpg")
            
            # 绘制
            self.screen.fill((0, 0, 0))
            stretched = self.draw_stretched_image()
            self.screen.blit(stretched, (0, 0))
            self.draw_control_points()
            
            if self.dragging:
                font = pygame.font.Font(None, 24)
                pos = self.control_points[self.dragging]
                text = font.render(f"{self.dragging.upper()}: ({int(pos[0])}, {int(pos[1])})", 
                                 True, self.RED)
                self.screen.blit(text, (10, 10))
            
            pygame.display.flip()
            clock.tick(60)  # 限制帧率为60fps

        pygame.quit()

if __name__ == "__main__":
    editor = ImagePerspectiveEditor("20.jpg")
    editor.run() 