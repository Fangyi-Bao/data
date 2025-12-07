import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 1. 数据预处理和加载
# 定义数据转换：将图像转换为张量并归一化
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量 (0-1范围)
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST的标准归一化参数
])

# 下载并加载MNIST数据集
train_dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
print(f"批次大小: {batch_size}")

# 2. 定义神经网络模型
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # 第一个全连接层: 784 (28x28) -> 128个神经元
        self.fc1 = nn.Linear(28 * 28, 128)
        # 第二个全连接层: 128 -> 64个神经元
        self.fc2 = nn.Linear(128, 64)
        # 输出层: 64 -> 10个类别 (0-9)
        self.fc3 = nn.Linear(64, 10)
        # Dropout层防止过拟合
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # 展平图像 (批次大小, 1, 28, 28) -> (批次大小, 784)
        x = x.view(-1, 28 * 28)
        # 第一层 + ReLU激活函数
        x = F.relu(self.fc1(x))
        # Dropout
        x = self.dropout(x)
        # 第二层 + ReLU激活函数
        x = F.relu(self.fc2(x))
        # Dropout
        x = self.dropout(x)
        # 输出层 (不需要softmax，因为CrossEntropyLoss会处理)
        x = self.fc3(x)
        return x

# 3. 初始化模型、损失函数和优化器
model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("模型结构:")
print(model)

# 4. 训练函数
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        
        # 计算损失
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 更新权重
        optimizer.step()
        
        # 统计
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 每100个批次打印一次进度
        if batch_idx % 100 == 0:
            print(f'训练轮次: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t损失: {loss.item():.6f}')
    
    # 计算平均损失和准确率
    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

# 5. 测试函数
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 测试时不需要计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'\n测试集: 平均损失: {avg_loss:.4f}, 准确率: {correct}/{total} ({accuracy:.2f}%)\n')
    
    return avg_loss, accuracy

# 6. 训练模型
epochs = 10
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

print("开始训练...")
for epoch in range(1, epochs + 1):
    # 训练
    train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # 测试
    test_loss, test_acc = test(model, device, test_loader, criterion)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    
    # 如果准确率超过95%，可以提前停止
    if test_acc > 95:
        print(f"测试准确率已达到{test_acc:.2f}%，提前停止训练！")
        break

# 7. 可视化训练过程
plt.figure(figsize=(12, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='训练损失')
plt.plot(range(1, len(test_losses) + 1), test_losses, 'r-', label='测试损失')
plt.xlabel('训练轮次')
plt.ylabel('损失')
plt.title('训练和测试损失')
plt.legend()
plt.grid(True)

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, 'b-', label='训练准确率')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, 'r-', label='测试准确率')
plt.xlabel('训练轮次')
plt.ylabel('准确率 (%)')
plt.title('训练和测试准确率')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 8. 在测试集上做最终评估
print("最终测试结果:")
model.eval()
correct = 0
total = 0
all_predictions = []
all_targets = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 保存预测结果用于进一步分析
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

final_accuracy = 100. * correct / total
print(f"最终测试准确率: {correct}/{total} ({final_accuracy:.2f}%)")

if final_accuracy > 90:
    print("✓ 达到目标: 测试准确率 > 90%")
else:
    print("✗ 未达到目标: 测试准确率 < 90%")