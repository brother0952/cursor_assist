import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class HousePriceDataset(Dataset):
    """
    自定义数据集类，用于加载房价数据
    继承自torch.utils.data.Dataset
    """
    def __init__(self, features, targets):
        """
        初始化数据集
        Args:
            features: 特征数据，形状为(n_samples, n_features)
            targets: 目标房价，形状为(n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets).reshape(-1, 1)
    
    def __len__(self):
        """返回数据集的样本数量"""
        return len(self.targets)
    
    def __getitem__(self, idx):
        """
        获取指定索引的样本
        Args:
            idx: 样本索引
        Returns:
            feature: 特征数据
            target: 对应的房价
        """
        return self.features[idx], self.targets[idx]

class HousePriceRegressor(nn.Module):
    """
    房价预测模型
    使用多层感知机进行回归预测
    """
    def __init__(self, input_size):
        """
        初始化模型结构
        Args:
            input_size: 输入特征的维度
        """
        super(HousePriceRegressor, self).__init__()
        # 构建一个三层的神经网络
        self.layers = nn.Sequential(
            # 第一层：输入层到第一个隐藏层
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),  # 批标准化
            nn.Dropout(0.2),
            
            # 第二层：第一个隐藏层到第二个隐藏层
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            
            # 第三层：第二个隐藏层到输出层
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入数据，形状为(batch_size, input_size)
        Returns:
            输出预测结果，形状为(batch_size, 1)
        """
        return self.layers(x)

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
        epoch_rmse: 均方根误差
    """
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    # 使用tqdm显示训练进度
    progress_bar = tqdm(train_loader, desc='训练中')
    for features, target in progress_bar:
        # 将数据移到指定设备
        features, target = features.to(device), target.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        output = model(features)
        
        # 计算损失
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 记录损失和预测结果
        total_loss += loss.item()
        predictions.extend(output.detach().cpu().numpy())
        targets.extend(target.detach().cpu().numpy())
        
        # 计算RMSE并更新进度条
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(targets)) ** 2))
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'rmse': f'{rmse:.4f}'
        })
    
    # 计算整个epoch的平均损失和RMSE
    epoch_loss = total_loss / len(train_loader)
    epoch_rmse = np.sqrt(np.mean((np.array(predictions) - np.array(targets)) ** 2))
    
    return epoch_loss, epoch_rmse

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
        test_rmse: 测试集上的均方根误差
    """
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    # 不计算梯度
    with torch.no_grad():
        for features, target in test_loader:
            features, target = features.to(device), target.to(device)
            output = model(features)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            predictions.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    # 计算平均损失和RMSE
    test_loss = total_loss / len(test_loader)
    test_rmse = np.sqrt(np.mean((np.array(predictions) - np.array(targets)) ** 2))
    
    return test_loss, test_rmse

def plot_training_history(train_losses, train_rmses, test_losses, test_rmses):
    """
    绘制训练历史曲线
    Args:
        train_losses: 训练损失历史
        train_rmses: 训练RMSE历史
        test_losses: 测试损失历史
        test_rmses: 测试RMSE历史
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
    
    # 绘制RMSE曲线
    ax2.plot(epochs, train_rmses, 'b-', label='训练RMSE')
    ax2.plot(epochs, test_rmses, 'r-', label='测试RMSE')
    ax2.set_title('训练和测试RMSE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('house_price_training_history.png')
    plt.close()

def plot_predictions(y_true, y_pred):
    """
    绘制预测值与真实值的散点图
    Args:
        y_true: 真实房价
        y_pred: 预测房价
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('真实房价')
    plt.ylabel('预测房价')
    plt.title('预测房价 vs 真实房价')
    plt.tight_layout()
    plt.savefig('house_price_predictions.png')
    plt.close()

def main():
    # 设置随机种子，确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载加利福尼亚房价数据集
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建数据集和数据加载器
    train_dataset = HousePriceDataset(X_train, y_train)
    test_dataset = HousePriceDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # 创建模型
    input_size = X.shape[1]  # 特征数量
    model = HousePriceRegressor(input_size).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练参数
    num_epochs = 100
    best_rmse = float('inf')
    
    # 用于记录训练历史
    train_losses = []
    train_rmses = []
    test_losses = []
    test_rmses = []
    
    print("开始训练...")
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        
        # 训练一个epoch
        train_loss, train_rmse = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f'训练集 - 损失: {train_loss:.4f}, RMSE: {train_rmse:.4f}')
        
        # 评估模型
        test_loss, test_rmse = evaluate(model, test_loader, criterion, device)
        print(f'测试集 - 损失: {test_loss:.4f}, RMSE: {test_rmse:.4f}')
        
        # 记录历史
        train_losses.append(train_loss)
        train_rmses.append(train_rmse)
        test_losses.append(test_loss)
        test_rmses.append(test_rmse)
        
        # 保存最佳模型
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            torch.save(model.state_dict(), 'house_price_model.pth')
            print(f'模型已保存，当前最佳RMSE: {best_rmse:.4f}')
    
    # 绘制训练历史
    plot_training_history(train_losses, train_rmses, test_losses, test_rmses)
    print("\n训练历史已保存到 house_price_training_history.png")
    
    # 加载最佳模型进行预测
    model.load_state_dict(torch.load('house_price_model.pth'))
    model.eval()
    
    # 在测试集上进行预测
    predictions = []
    targets = []
    with torch.no_grad():
        for features, target in test_loader:
            features = features.to(device)
            output = model(features)
            predictions.extend(output.cpu().numpy())
            targets.extend(target.numpy())
    
    # 将标准化的预测值转换回原始尺度
    predictions = scaler_y.inverse_transform(np.array(predictions))
    targets = scaler_y.inverse_transform(np.array(targets))
    
    # 绘制预测结果
    plot_predictions(targets.ravel(), predictions.ravel())
    print("预测结果可视化已保存到 house_price_predictions.png")
    
    # 计算并打印最终的评估指标
    final_rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    final_mae = np.mean(np.abs(predictions - targets))
    final_r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - targets.mean()) ** 2)
    
    print("\n最终评估结果:")
    print(f"RMSE: ${final_rmse:.2f}k")
    print(f"MAE: ${final_mae:.2f}k")
    print(f"R² Score: {final_r2:.4f}")

if __name__ == '__main__':
    main() 