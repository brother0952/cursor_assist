import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

class COCODetectionDataset:
    def __init__(self, root, split='train', transform=None):
        self.transform = transform
        self.dataset = torchvision.datasets.CocoDetection(
            root=root,
            annFile=f'{root}/annotations/instances_{split}2017.json'
        )
        # COCO类别ID到名称的映射
        self.categories = {
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
            6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light'
        }  # 这里只列出了部分类别，完整列表请参考COCO官方文档
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, annotations = self.dataset[idx]
        boxes = []
        labels = []
        
        # 提取边界框和标签
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            # 转换为[x1, y1, x2, y2]格式
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            boxes.append(bbox)
            labels.append(ann['category_id'])
        
        # 转换为张量
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
        }
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target

def get_transform():
    transforms = []
    # 转换为张量
    transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(transforms)

def train_model(model, data_loader, optimizer, device, num_epochs=10):
    model.train()
    print("开始训练...")
    
    for epoch in range(num_epochs):
        print(f"第 {epoch+1} 轮训练")
        running_loss = 0.0
        
        for i, (images, targets) in enumerate(data_loader):
            # 将数据移到设备
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # 前向传播
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # 反向传播
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            running_loss += losses.item()
            
            if i % 10 == 9:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
                running_loss = 0.0
    
    print("训练完成！")

def visualize_detection(model, image_path, device, confidence_threshold=0.5):
    """可视化目标检测结果"""
    model.eval()
    
    # 加载和预处理图像
    image = Image.open(image_path).convert('RGB')
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # 获取预测结果
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    
    # 创建图像
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    # 绘制检测框
    for box, score, label in zip(boxes, scores, labels):
        if score > confidence_threshold:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(rect)
            plt.text(
                x1, y1,
                f'Class {label}: {score:.2f}',
                bbox=dict(facecolor='white', alpha=0.8)
            )
    
    plt.axis('off')
    plt.savefig('detection_result.png')
    plt.close()

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据集路径（请替换为实际的COCO数据集路径）
    data_path = './coco'
    
    # 创建数据集和数据加载器
    dataset = COCODetectionDataset(
        root=data_path,
        split='train',
        transform=get_transform()
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # 加载预训练模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    
    # 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # 训练模型
    train_model(model, data_loader, optimizer, device)
    
    # 保存模型
    torch.save(model.state_dict(), 'object_detection_model.pth')
    
    # 测试检测效果
    test_image_path = 'test_image.jpg'  # 请替换为实际的测试图像路径
    visualize_detection(model, test_image_path, device)
    
    print("模型已保存，可视化结果已生成。")

if __name__ == '__main__':
    main() 