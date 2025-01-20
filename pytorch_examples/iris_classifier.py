import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class IrisDataset(Dataset):
    """
    自定义数据集类，用于加载鸢尾花数据
    继承自torch.utils.data.Dataset
    """
    def __init__(self, features, labels):
        """
        初始化数据集
        Args:
            features: 特征数据，形状为(n_samples, n_features)
            labels: 标签数据，形状为(n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        """返回数据集的样本数量"""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        获取指定索引的样本
        Args:
            idx: 样本索引
        Returns:
            feature: 特征数据
            label: 对应的标签
        """
        return self.features[idx], self.labels[idx]

class IrisClassifier(nn.Module):
    """
    鸢尾花分类模型
    使用简单的前馈神经网络进行分类
    """
    def __init__(self, input_size, hidden_size, num_classes):
        """
        初始化模型结构
        Args:
            input_size: 输入特征的维度
            hidden_size: 隐藏层的神经元数量
            num_classes: 分类类别数量
        """
        super(IrisClassifier, self).__init__()
        # 第一个全连接层，从输入层到隐藏层
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 第二个全连接层，从隐藏层到输出层
        self.fc2 = nn.Linear(hidden_size, num_classes)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入数据，形状为(batch_size, input_size)
        Returns:
            输出预测结果，形状为(batch_size, num_classes)
        """
        # 第一层：全连接 + ReLU + Dropout
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # 第二层：全连接
        x = self.fc2(x)
        return x

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练一个epoch
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备（CPU/GPU）
    Returns:
        epoch_loss: 平均损失
        epoch_acc: 平均准确率
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # 使用tqdm显示训练进度
    progress_bar = tqdm(train_loader, desc='训练中')
    for features, labels in progress_bar:
        # 将数据移到指定设备
        features, labels = features.to(device), labels.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(features)
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 统计损失和准确率
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 更新进度条信息
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    # 计算平均损失和准确率
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def evaluate(model, test_loader, criterion, device):
    """
    评估模型性能
    Args:
        model: 模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 计算设备（CPU/GPU）
    Returns:
        test_loss: 测试集上的平均损失
        test_acc: 测试集上的准确率
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # 不计算梯度
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # 计算平均损失和准确率
    test_loss = total_loss / len(test_loader)
    test_acc = correct / total
    
    return test_loss, test_acc

def plot_training_history(train_losses, train_accs, test_losses, test_accs):
    """
    绘制训练历史曲线
    Args:
        train_losses: 训练损失历史
        train_accs: 训练准确率历史
        test_losses: 测试损失历史
        test_accs: 测试准确率历史
    """
    epochs = range(1, len(train_losses) + 1)
    
    # 创建一个包含两个子图的图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 绘制损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='训练损失')
    ax1.plot(epochs, test_losses, 'r-', label='测试损失')
    ax1.set_title('训练和测试损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失')
    ax1.legend()
    
    # 绘制准确率曲线
    ax2.plot(epochs, train_accs, 'b-', label='训练准确率')
    ax2.plot(epochs, test_accs, 'r-', label='测试准确率')
    ax2.set_title('训练和测试准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('准确率')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('iris_training_history.png')
    plt.close()

def main():
    # 设置随机种子，确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载鸢尾花数据集
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建数据集和数据加载器
    train_dataset = IrisDataset(X_train, y_train)
    test_dataset = IrisDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 创建模型
    input_size = 4  # 鸢尾花数据集有4个特征
    hidden_size = 10  # 隐藏层神经元数量
    num_classes = 3  # 鸢尾花有3个类别
    
    model = IrisClassifier(input_size, hidden_size, num_classes).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 训练参数
    num_epochs = 100
    best_accuracy = 0
    
    # 用于记录训练历史
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    print("开始训练...")
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        
        # 训练一个epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f'训练集 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}')
        
        # 评估模型
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f'测试集 - 损失: {test_loss:.4f}, 准确率: {test_acc:.4f}')
        
        # 记录历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'iris_model.pth')
            print(f'模型已保存，当前最佳准确率: {best_accuracy:.4f}')
    
    # 绘制训练历史
    plot_training_history(train_losses, train_accs, test_losses, test_accs)
    print("\n训练历史已保存到 iris_training_history.png")
    
    # 打印最终结果
    print(f"\n训练完成！最佳测试准确率: {best_accuracy:.4f}")
    
    # 加载最佳模型进行预测示例
    model.load_state_dict(torch.load('iris_model.pth'))
    model.eval()
    
    # 准备一些测试数据
    test_samples = torch.FloatTensor(X_test[:5]).to(device)
    with torch.no_grad():
        outputs = model(test_samples)
        _, predicted = torch.max(outputs.data, 1)
    
    # 打印预测结果
    print("\n预测示例:")
    for i in range(5):
        true_label = y_test[i]
        pred_label = predicted[i].item()
        print(f"样本 {i+1} - 真实类别: {iris.target_names[true_label]}, "
              f"预测类别: {iris.target_names[pred_label]}")

if __name__ == '__main__':
    main() 