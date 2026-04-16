class SimpleCNN(nn.Module):
    """简单的卷积神经网络模型，用于MNIST手写数字分类
  
    结构：一个卷积层 -> ReLU激活 -> 最大池化 -> 全连接层
    """
    def __init__(self):
        """初始化网络结构，定义各层组件"""
        super(SimpleCNN, self).__init__()
        # 卷积层：输入1通道(灰度图)，输出16通道，3x3卷积核，步长1，填充1
        self.conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # ReLU激活函数：增加非线性
        self.relu = nn.ReLU()
        # 最大池化层：2x2窗口，步长2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层：将特征映射到10个类别(数字0-9)
        # 输入维度计算：16(通道数) * 14(池化后高度) * 14(池化后宽度) = 3136
        self.fc = nn.Linear(in_features=16 * 14 * 14, out_features=10)

    def forward(self, x):
        """定义前向传播过程
      
        参数:
            x: 输入张量，形状为[batch_size, 1, 28, 28]
          
        返回:
            输出张量，形状为[batch_size, 10]
        """
        # 卷积 -> ReLU激活 -> 最大池化
        x = self.pool(self.relu(self.conv(x)))
        # 将特征图展平
        x = x.view(-1, 16 * 14 * 14)
        # 全连接层
        x = self.fc(x)
        return x这段代码定义了我们的 SimpleCNN类，它继承自 nn.Module(PyTorch中所有神经网络的基类)。让我们详细解释每一部分：1. 初始化函数(__init__)：• 定义了一个卷积层(self.conv)，将1通道的输入图像转换为16通道的特征图• 定义了ReLU激活函数(self.relu)，为网络引入非线性• 定义了最大池化层(self.pool)，减小特征图的空间维度• 定义了全连接层(self.fc)，将特征映射到10个类别2. 前向传播函数(forward)：• 实现了数据在网络中的流动过程• 首先通过卷积层、ReLU激活函数和池化层• 然后将特征图展平为一维向量• 最后通过全连接层得到最终输出4.3. 数据加载与预处理def load_data(batch_size=64):
    """加载MNIST数据集并进行预处理
  
    参数:
        batch_size: 批量大小，默认为64
      
    返回:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # 定义数据变换：转换为张量并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量，并将像素值归一化到[0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # 使用MNIST数据集的均值和标准差进行归一化
    ])

    # 加载训练集
    train_dataset = torchvision.datasets.MNIST(
        root='./data',  # 数据存储路径
        train=True,  # 指定为训练集
        download=True,  # 如果数据不存在，则下载
        transform=transform  # 应用上面定义的变换
    )

    # 加载测试集
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True# 打乱数据，增加随机性
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False# 测试集不需要打乱
    )

    return train_loader, test_loader
def train(model, train_loader, criterion, optimizer, device, epochs=5):
    """训练模型函数
  
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备(CPU或GPU)
        epochs: 训练轮数，默认为5
      
    返回:
        train_losses: 每个epoch的训练损失列表
    """
    model.train()  # 设置为训练模式
    train_losses = []  # 存储每个epoch的平均损失

    for epoch inrange(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
      
        for i, (images, labels) inenumerate(train_loader):
            # 将数据移动到指定设备
            images, labels = images.to(device), labels.to(device)
          
            # 清零梯度
            optimizer.zero_grad()
          
            # 前向传播
            outputs = model(images)
          
            # 计算损失
            loss = criterion(outputs, labels)
          
            # 反向传播
            loss.backward()
          
            # 更新参数
            optimizer.step()
          
            # 累加损失
            running_loss += loss.item()
          
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
          
        # 计算每个epoch的平均损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
      
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    return train_losses
def test(model, test_loader, criterion, device):
    """测试模型函数
  
    参数:
        model: 要测试的模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 计算设备(CPU或GPU)
      
    返回:
        test_loss: 测试损失
        accuracy: 测试准确率
    """
    model.eval()  # 设置为评估模式
    test_loss = 0.0
    correct = 0
    total = 0

    # 不计算梯度，节省内存
    with torch.no_grad():
        for images, labels in test_loader:
            # 将数据移动到指定设备
            images, labels = images.to(device), labels.to(device)
          
            # 前向传播
            outputs = model(images)
          
            # 计算损失
            loss = criterion(outputs, labels)
            test_loss += loss.item()
          
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算平均损失和准确率
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total

    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return test_loss, accuracy
def display_data(data_loader, num_images=25):
    """显示数据集中的图像和标签
  
    参数:
        data_loader: 数据加载器
        num_images: 要显示的图像数量，默认为25
    """
    # 获取一批数据
    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    # 计算网格大小
    grid_size = int(np.ceil(np.sqrt(num_images)))

    # 创建图形
    plt.figure(figsize=(10, 10))

    # 显示图像
    for i inrange(min(num_images, images.shape[0])):
        plt.subplot(grid_size, grid_size, i+1)
        # 转换图像格式：[C,H,W] -> [H,W]，并反归一化
        img = images[i][0].cpu().numpy()
        plt.imshow(img, cmap='gray')
        plt.title(f'标签: {labels[i]}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('mnist_samples.png')
    plt.show()
def visualize_predictions(model, test_loader, device, num_images=5):
    """可视化模型预测结果
  
    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备(CPU或GPU)
        num_images: 要显示的图像数量，默认为5
    """
    model.eval()

    # 获取一批数据
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # 将图像移动到指定设备并进行预测
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # 将图像移回CPU用于显示
    images = images.cpu()

    # 创建图形
    plt.figure(figsize=(12, 4))

    # 显示图像和预测结果
    for i inrange(min(num_images, images.shape[0])):
        plt.subplot(1, num_images, i+1)
        img = images[i][0].numpy()
        plt.imshow(img, cmap='gray')
        color = 'green' if predicted[i] == labels[i] else'red'
        plt.title(f'预测: {predicted[i]}\n真实: {labels[i]}', color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()
