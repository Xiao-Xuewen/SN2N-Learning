import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

num_classes = 5
num_per_class = 100
num_samples = num_classes * num_per_class
noise = 0.2
seed = 0
input_dim = 2
hidden_dim_1 = 32
hidden_dim_2 = 16
output_dim = num_classes
learning_rate = 0.1 #要看参数更新次数
num_steps = 5000

num_epoch = 2000
batch_size = 32

loss_history = []
acc_history = []

def make_2D_spiral_data(num_classes,num_per_class,noise,seed):
    num_samples = num_classes * num_per_class
    X = torch.zeros((num_samples,2),dtype=torch.float32)
    y = torch.zeros(num_samples,dtype=torch.long)
    generator = torch.Generator()
    generator.manual_seed(seed)
    for j in range(num_classes):
        idx = slice(j*num_per_class,(j+1)*num_per_class)
        r = torch.linspace(0,1,num_per_class)
        theta = torch.linspace(j*4,(j+1)*4,num_per_class)
        theta += torch.randn(num_per_class,generator=generator)*noise
        X[idx,0] = r*torch.sin(theta)
        X[idx,1] = r*torch.cos(theta)
        y[idx] = j
    return X,y

X,y = make_2D_spiral_data(num_classes,num_per_class,noise,seed)
dataset = TensorDataset(X,y)
train_loader = DataLoader(dataset,batch_size,shuffle=True)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim_1,hidden_dim_2,output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1,hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2,output_dim)
    def forward(self,x):
        z1 = self.fc1(x)
        a1 = torch.relu(z1)
        z2 = self.fc2(a1)
        a2 = torch.relu(z2)
        logits = self.fc3(a2)
        return logits
torch.manual_seed(seed)
model = MLP(input_dim,hidden_dim_1,hidden_dim_2,output_dim)
# 绘图代码由ChatGPT生成
initial_state = {
    name: tensor.detach().clone()
    for name, tensor in model.state_dict().items()
}
# 绘图代码由ChatGPT生成
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)


for epoch in range(num_epoch):
    running_loss = 0.0
    correct = 0
    total = 0
    for train_X,train_y in train_loader:
        logits = model(train_X)
        loss = criterion(logits,train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_batch_size = train_y.size(0)
        running_loss += (loss.item() * current_batch_size)
        prediction = torch.argmax(logits,dim = 1)
        correct += (train_y == prediction).sum().item()
        total += current_batch_size
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    loss_history.append(epoch_loss)
    acc_history.append(epoch_acc)

    if epoch % 100 == 0:
        print(f"epoch: {epoch:4d}")
        print(f"loss: {epoch_loss:.4f}")
        print(f"acc: {epoch_acc:.4f}")

model.eval()
with torch.no_grad():
    logits = model(X)
    loss = criterion(logits,y)
    prediction = torch.argmax(logits,dim = 1)
    acc = torch.mean((y == prediction).float())
    print(f"Final acc:{acc:.4f}")

# 绘图代码由ChatGPT生成

# 创建一个拥有“训练前参数”的模型
model_before = MLP(input_dim, hidden_dim_1, hidden_dim_2, output_dim)
model_before.load_state_dict(initial_state)
model_before.eval()


def predict_grid(model, X, grid_size=300):
    """在二维平面网格上预测类别，用于画决策边界。"""

    x_min = X[:, 0].min().item() - 0.2
    x_max = X[:, 0].max().item() + 0.2
    y_min = X[:, 1].min().item() - 0.2
    y_max = X[:, 1].max().item() + 0.2

    x_values = torch.linspace(x_min, x_max, grid_size)
    y_values = torch.linspace(y_min, y_max, grid_size)

    xx, yy = torch.meshgrid(
        x_values,
        y_values,
        indexing="xy"
    )

    # 每一行都是一个二维坐标点，shape = (grid_size * grid_size, 2)
    grid = torch.stack(
        (xx.reshape(-1), yy.reshape(-1)),
        dim=1
    )

    with torch.no_grad():
        logits = model(grid)
        prediction = torch.argmax(logits, dim=1)
        prediction = prediction.reshape(xx.shape)

    return xx, yy, prediction


# 训练前与训练后的决策区域
xx_before, yy_before, pred_before = predict_grid(model_before, X)
xx_after, yy_after, pred_after = predict_grid(model, X)

# Matplotlib 绘图时转为 NumPy 数组
X_plot = X.detach().cpu().numpy()
y_plot = y.detach().cpu().numpy()

xx_before = xx_before.cpu().numpy()
yy_before = yy_before.cpu().numpy()
pred_before = pred_before.cpu().numpy()

xx_after = xx_after.cpu().numpy()
yy_after = yy_after.cpu().numpy()
pred_after = pred_after.cpu().numpy()


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 训练前
axes[0].contourf(
    xx_before,
    yy_before,
    pred_before,
    alpha=0.3
)
axes[0].scatter(
    X_plot[:, 0],
    X_plot[:, 1],
    c=y_plot
)
axes[0].set_title("Before Training")
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")

# 训练后
axes[1].contourf(
    xx_after,
    yy_after,
    pred_after,
    alpha=0.3
)
axes[1].scatter(
    X_plot[:, 0],
    X_plot[:, 1],
    c=y_plot
)
axes[1].set_title(f"After Training (acc={acc.item():.4f})")
axes[1].set_xlabel("x1")
axes[1].set_ylabel("x2")

plt.tight_layout()

plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.figure()
plt.plot(acc_history)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.show()

# 绘图代码由ChatGPT生成


        

    
