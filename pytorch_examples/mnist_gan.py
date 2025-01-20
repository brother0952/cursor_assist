import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 生成器网络
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # 输入维度: latent_dim
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, 784),  # 28x28=784
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img

# 判别器网络
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # 输入维度: 784 (28x28)
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

def train_gan(generator, discriminator, dataloader, num_epochs, latent_dim, device):
    # 损失函数和优化器
    adversarial_loss = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # 训练循环
    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            
            # 创建标签
            real = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)
            
            # 真实图像
            real_imgs = imgs.to(device)
            
            # 训练判别器
            d_optimizer.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), real)
            
            # 生成假图像
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z)
            fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake)
            
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()
            
            # 训练生成器
            g_optimizer.zero_grad()
            g_loss = adversarial_loss(discriminator(fake_imgs), real)
            g_loss.backward()
            g_optimizer.step()
            
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
        
        # 每个epoch保存生成的图像样本
        if (epoch + 1) % 5 == 0:
            save_sample_images(generator, epoch + 1, latent_dim, device)

def save_sample_images(generator, epoch, latent_dim, device, n_samples=16):
    """保存生成的样本图像"""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim).to(device)
        gen_imgs = generator(z).cpu()
    
    # 将图像转换为numpy数组并重新缩放
    gen_imgs = gen_imgs.numpy()
    gen_imgs = 0.5 * gen_imgs + 0.5
    
    fig, axs = plt.subplots(4, 4)
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(gen_imgs[i*4 + j, 0, :, :], cmap='gray')
            axs[i, j].axis('off')
    
    plt.savefig(f'gan_samples_epoch_{epoch}.png')
    plt.close()
    generator.train()

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 超参数
    latent_dim = 100
    batch_size = 64
    num_epochs = 200
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # 加载MNIST数据集
    print("加载数据集...")
    dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    # 创建生成器和判别器
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    
    print("开始训练...")
    train_gan(generator, discriminator, dataloader, num_epochs, latent_dim, device)
    
    # 保存模型
    torch.save(generator.state_dict(), 'mnist_generator.pth')
    torch.save(discriminator.state_dict(), 'mnist_discriminator.pth')
    print("训练完成！模型已保存。")

if __name__ == '__main__':
    main() 