import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root='./assignment/minst_mlp/data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./assignment/minst_mlp/data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

images,labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)
print(images.dtype)
print(labels.dtype)
print(labels[:10])

input_dim = 28*28
hidden_dim_1 = 128
hidden_dim_2 = 128
output_dim = 10
learning_rate = 0.1
num_epoch = 10

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim_1,hidden_dim_2,output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1,hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2,output_dim)
    def forward(self,x):
        x = torch.flatten(x,start_dim= 1)
        z1 = self.fc1(x)
        a1 = torch.relu(z1)
        z2 = self.fc2(a1)
        a2 = torch.relu(z2)
        logits = self.fc3(a2)
        return logits
    
criterion= nn.CrossEntropyLoss()
model = MLP(input_dim, hidden_dim_1,hidden_dim_2,output_dim)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate
)

# 绘图代码由ChatGPT生成
train_loss_history = []
train_acc_history = []

test_loss_history = []
test_acc_history = []
# 绘图代码由ChatGPT生成

for epoch in range(num_epoch):

    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    for train_x,train_y in train_loader:

        logits = model(train_x)
        loss = criterion(logits,train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_batch_size = train_y.size(0)
        prediction = torch.argmax(logits,dim = 1)
        correct += (prediction == train_y).sum().item()
        total += current_batch_size
        running_loss +=  current_batch_size * loss.item()
    train_acc = correct / total
    train_loss = running_loss / total

    model.eval()
    with torch.no_grad():
        correct_test = 0
        total_test = 0
        running_loss_test = 0.0
        for test_x,test_y in test_loader:

            logits = model(test_x)
            loss = criterion(logits,test_y)

            current_batch_size = test_y.size(0)
            prediction = torch.argmax(logits,dim = 1)
            correct_test += (prediction == test_y).sum().item()
            total_test += current_batch_size
            running_loss_test +=  current_batch_size * loss.item()
        test_acc = correct_test / total_test
        test_loss = running_loss_test / total_test
    # 绘图代码由ChatGPT生成
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)

    test_loss_history.append(test_loss)
    test_acc_history.append(test_acc)
    # 绘图代码由ChatGPT生成
    print(
        f"epoch {epoch:2d} | "
        f"train loss {train_loss:.4f} | "
        f"train acc {train_acc:.4f} | "
        f"test loss {test_loss:.4f} | "
        f"test acc {test_acc:.4f}"
        )

# 绘图代码由ChatGPT生成
# ------------------------------
# 1. Loss 曲线
# ------------------------------

epochs = range(1, num_epoch + 1)

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_loss_history,
    label='Train Loss'
)

plt.plot(
    epochs,
    test_loss_history,
    label='Test Loss'
)

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.title('Training and Test Loss')

plt.legend()

plt.grid()




# ------------------------------
# 2. Accuracy 曲线
# ------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_acc_history,
    label='Train Accuracy'
)

plt.plot(
    epochs,
    test_acc_history,
    label='Test Accuracy'
)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.title('Training and Test Accuracy')

plt.legend()

plt.grid()




# ------------------------------
# 3. 查看模型预测结果
# ------------------------------

model.eval()

with torch.no_grad():

    test_images, test_labels = next(iter(test_loader))

    logits = model(test_images)

    predictions = torch.argmax(
        logits,
        dim=1
    )


# ------------------------------
# 4. 显示前 16 张图片及预测结果
# ------------------------------

plt.figure(figsize=(10, 10))

for i in range(16):

    plt.subplot(4, 4, i + 1)

    plt.imshow(
        test_images[i].squeeze(),
        cmap='gray'
    )

    plt.title(
        f"True: {test_labels[i].item()} | "
        f"Pred: {predictions[i].item()}"
    )

    plt.axis('off')

plt.suptitle('MNIST Predictions')

plt.tight_layout()




# ------------------------------
# 5. 显示预测错误的图片
# ------------------------------

wrong_indices = (
    predictions != test_labels
).nonzero().squeeze()


if wrong_indices.numel() > 0:

    wrong_indices = wrong_indices.flatten()

    num_wrong_to_show = min(
        16,
        wrong_indices.numel()
    )

    plt.figure(figsize=(10, 10))

    for i in range(num_wrong_to_show):

        index = wrong_indices[i].item()

        plt.subplot(4, 4, i + 1)

        plt.imshow(
            test_images[index].squeeze(),
            cmap='gray'
        )

        plt.title(
            f"True: {test_labels[index].item()} | "
            f"Pred: {predictions[index].item()}"
        )

        plt.axis('off')

    plt.suptitle('Wrong Predictions')

    plt.tight_layout()

    plt.show()

else:
    
    plt.show()
    print("No wrong predictions in this batch.")
# 绘图代码由ChatGPT生成
