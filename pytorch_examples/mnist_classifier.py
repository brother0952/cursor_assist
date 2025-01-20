import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一个卷积层，1个输入通道，32个输出通道，3x3卷积核
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 第二个卷积层，32个输入通道，64个输出通道，3x3卷积核
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 全连接层
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 第一个卷积层 + ReLU + 最大池化
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        # 第二个卷积层 + ReLU + 最大池化
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # 展平张量
        x = x.view(-1, 7 * 7 * 64)
        # 第一个全连接层 + ReLU + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # 第二个全连接层
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='训练中')
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    return total_loss / len(train_loader), correct / total

def evaluate(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    
    return test_loss, accuracy

def visualize_predictions(model, test_loader, device, num_images=10):
    model.eval()
    images, labels = next(iter(test_loader))
    images = images[:num_images]
    labels = labels[:num_images]
    
    with torch.no_grad():
        outputs = model(images.to(device))
        predictions = outputs.max(1, keepdim=True)[1]
    
    fig = plt.figure(figsize=(15, 3))
    for i in range(num_images):
        ax = fig.add_subplot(1, num_images, i + 1)
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title(f'预测: {predictions[i].item()}\n实际: {labels[i].item()}')
        ax.axis('off')
    
    plt.savefig('mnist_predictions.png')
    plt.close()

def main():
    # 设置随机种子，确保结果可复现
    torch.manual_seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])
    
    # 加载MNIST数据集
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    
    # 创建模型
    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 10
    best_accuracy = 0
    
    print("开始训练...")
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        
        # 训练一个epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        print(f'训练集 - 平均损失: {train_loss:.4f}, 准确率: {train_acc:.4f}')
        
        # 评估模型
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f'测试集 - 平均损失: {test_loss:.4f}, 准确率: {test_acc:.4f}')
        
        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'mnist_model.pth')
            print(f'模型已保存，当前最佳准确率: {best_accuracy:.4f}')
    
    # 加载最佳模型并可视化预测结果
    model.load_state_dict(torch.load('mnist_model.pth'))
    print("\n可视化预测结果...")
    visualize_predictions(model, test_loader, device)
    print("预测结果已保存到 mnist_predictions.png")

if __name__ == '__main__':
    main() 