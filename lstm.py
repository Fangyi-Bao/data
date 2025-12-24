# ==============================
# 0. 导入库
# ==============================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 固定随机种子（可复现）
torch.manual_seed(42)
np.random.seed(42)


# ==============================
# 1. 构造原始数据集（线性趋势 + 季节性 + 噪声）
# ==============================
n_points = 1000
time_steps = np.linspace(0, 20, n_points)

# 组合多个成分：线性趋势 + 季节性 + 噪声
linear_trend = 0.1 * time_steps  # 线性趋势
seasonal = 2 * np.sin(2 * np.pi * time_steps)  # 季节性成分
noise = 0.5 * np.random.randn(n_points)  # 随机噪声

data = linear_trend + seasonal + noise

# 可视化原始数据
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time_steps, data, label='组合数据')
plt.plot(time_steps, linear_trend, 'r--', label='线性趋势', alpha=0.7)
plt.title("原始时间序列数据（线性趋势 + 季节性 + 噪声）")
plt.xlabel("时间")
plt.ylabel("数值")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(time_steps, seasonal, 'g--', label='季节性成分', alpha=0.7)
plt.plot(time_steps, noise, 'k:', label='噪声', alpha=0.5)
plt.xlabel("时间")
plt.ylabel("数值")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ==============================
# 2. 滑动窗口：原始数据 → 样本
# ==============================
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i + seq_len])
        ys.append(data[i + seq_len])
    return np.array(xs), np.array(ys)


SEQ_LEN = 20
X, y = create_sequences(data, SEQ_LEN)

# ==============================
# 3. 转成 LSTM 需要的张量格式
# ==============================
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 增加特征维度（input_size = 1）
X = X.unsqueeze(-1)   # (samples, seq_len, 1)
y = y.unsqueeze(-1)   # (samples, 1)

# ==============================
# 4. 划分训练 / 测试集
# ==============================
train_size = int(len(X) * 0.8)

X_train = X[:train_size]
y_train = y[:train_size]

X_test = X[train_size:]
y_test = y[train_size:]


# ==============================
# 5. 定义 LSTM 模型
# ==============================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步
        return out


# ==============================
# 6. 初始化模型 & 训练参数
# ==============================
model = LSTMModel(
    input_size=1,
    hidden_size=64,
    num_layers=2,  # 增加层数以提高模型容量
    output_size=1
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ==============================
# 7. 训练模型
# ==============================
EPOCHS = 200
loss_history = []
model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")


# ==============================
# 8. 测试 & 预测
# ==============================
model.eval()
with torch.no_grad():
    predictions = model(X_test)

# 转成 numpy 方便画图
predictions = predictions.squeeze().numpy()
y_test_np = y_test.squeeze().numpy()


# ==============================
# 9. 可视化预测结果
# ==============================
plt.figure(figsize=(12, 8))

# 子图1：完整数据集的预测结果
plt.subplot(3, 1, 1)
test_indices = np.arange(len(data))[-len(y_test_np):]
plt.plot(test_indices, y_test_np, label="真实值", linewidth=2)
plt.plot(test_indices, predictions, label="预测值", linewidth=1.5, linestyle='--')
plt.xlabel("时间步")
plt.ylabel("数值")
plt.title("测试集预测结果对比")
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2：前100个测试点的详细对比
plt.subplot(3, 1, 2)
plt.plot(test_indices[:100], y_test_np[:100], label="真实值", marker='o', markersize=4, linewidth=2)
plt.plot(test_indices[:100], predictions[:100], label="预测值", marker='s', markersize=4, linewidth=1.5)
plt.xlabel("时间步")
plt.ylabel("数值")
plt.title("前100个测试点详细对比")
plt.legend()
plt.grid(True, alpha=0.3)

# 子图3：残差分析
plt.subplot(3, 1, 3)
residuals = y_test_np - predictions
plt.plot(test_indices, residuals, label="残差", color='red', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.xlabel("时间步")
plt.ylabel("残差值")
plt.title("预测残差（真实值 - 预测值）")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ==============================
# 10. 可视化 Loss 曲线
# ==============================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("训练损失曲线")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# 最后50个epoch的损失
if len(loss_history) > 50:
    plt.plot(range(len(loss_history)-50, len(loss_history)), loss_history[-50:])
    plt.xlabel("最后50个Epoch")
    plt.ylabel("MSE Loss")
    plt.title("最后50个Epoch的训练损失")
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ==============================
# 11. 评估指标
# ==============================
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test_np, predictions)
mae = mean_absolute_error(y_test_np, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_np, predictions)

print("\n" + "="*50)
print("模型性能评估:")
print("="*50)
print(f"均方误差 (MSE): {mse:.6f}")
print(f"均方根误差 (RMSE): {rmse:.6f}")
print(f"平均绝对误差 (MAE): {mae:.6f}")
print(f"决定系数 (R²): {r2:.6f}")
print("="*50)