import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# 1. 定义 LeNet-5 模型结构
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 卷积层 1：输入 1x28x28，输出 6x28x28 (为了保持尺寸，padding设为2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        # 池化层 1：输出 6x14x14
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 卷积层 2：输入 6x14x14，输出 16x10x10
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 池化层 2：输出 16x5x5
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # 展平特征图
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)             # 输出层不加激活，配合 CrossEntropyLoss
        return x

# 2. 训练与测试函数
def train_and_evaluate():
    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 超参数
    batch_size = 64
    learning_rate = 0.001
    epochs = 5

    # 数据准备 (自动下载并转换为 Tensor，做标准化)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST 的均值和标准差
    ])

    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 开始训练
    print("\n--- Starting Training ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()           # 清空梯度
            outputs = model(images)         # 前向传播
            loss = criterion(outputs, labels) # 计算损失
            loss.backward()                 # 反向传播
            optimizer.step()                # 更新参数
            
            running_loss += loss.item()
            
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Time: {epoch_time:.2f}s")

    # 开始测试
    print("\n--- Starting Evaluation ---")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(): # 测试阶段不需要计算梯度
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy on 10000 test images: {accuracy:.2f}%")
    
    # 打印参数量，方便写报告
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params}")

if __name__ == '__main__':
    train_and_evaluate()
