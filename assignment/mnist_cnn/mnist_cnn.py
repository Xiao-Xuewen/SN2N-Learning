import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

# 超参数
input_channels = 1
image_size = 28
num_classes = 10

seed = 0

batch_size = 64
num_epoch = 20
learning_rate = 0.01

conv1_out_channels = 8
conv2_out_channels = 16

kernel_size = 3
padding = 1

pool_kernel_size = 2
pool_stride = 2

fc_hidden_dim = 64
feature_map_size = image_size // pool_stride // pool_stride

fc_input_dim = (
    conv2_out_channels
    * feature_map_size
    * feature_map_size
)

torch.manual_seed(seed)
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root='./assignment/mnist_cnn/data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./assignment/mnist_cnn/data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels = input_channels,
            out_channels = conv1_out_channels,
            kernel_size = kernel_size,
            padding = padding
        )
        self.conv2 = nn.Conv2d(
            in_channels = conv1_out_channels,
            out_channels = conv2_out_channels,
            kernel_size = kernel_size,
            padding = padding
        )
        self.pool = nn.MaxPool2d(
            kernel_size = pool_kernel_size,
            stride = pool_stride
        )
        self.fc1 = nn.Linear(fc_input_dim,fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim,num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = torch.relu(x)

        logits = self.fc2(x)

        return logits

model = CNN()

images, labels = next(iter(train_loader))

logits = model(images)

print(images.shape)
print(logits.shape)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr = learning_rate
)


train_loss_history = []
train_acc_history = []

test_loss_history = []
test_acc_history = []


epoch_time_history = []

training_start_time = time.perf_counter()

for epoch in range(num_epoch):

    epoch_start_time = time.perf_counter()
    model.train()

    correct_train = 0 
    total_train = 0
    running_loss_train = 0.0

    for x_train,y_train in train_loader:

        logits = model(x_train)
        loss = criterion(logits,y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_batch_size = y_train.size(0)

        prediction_train = torch.argmax(logits,dim = 1)
        correct_train += (y_train == prediction_train).sum().item()
        running_loss_train += loss.item() * current_batch_size
        total_train += current_batch_size

    train_acc =  correct_train / total_train
    train_loss = running_loss_train / total_train

    model.eval()
    with torch.no_grad():

        correct_test = 0 
        total_test = 0
        running_loss_test = 0.0

        for x_test,y_test in test_loader:

            logits = model(x_test)
            loss = criterion(logits,y_test)

            current_batch_size = y_test.size(0)

            prediction_test = torch.argmax(logits,dim = 1)
            correct_test += (y_test == prediction_test).sum().item()
            running_loss_test += loss.item() * current_batch_size
            total_test += current_batch_size
            
        test_acc =  correct_test / total_test
        test_loss = running_loss_test / total_test

    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)

    test_loss_history.append(test_loss)
    test_acc_history.append(test_acc)

    epoch_end_time = time.perf_counter()

    epoch_time = epoch_end_time - epoch_start_time

    epoch_time_history.append(epoch_time)
    
    print(
        f"epoch {epoch:2d} | "
        f"train loss {train_loss:.4f} | "
        f"train acc {train_acc:.4f} | "
        f"test loss {test_loss:.4f} | "
        f"test acc {test_acc:.4f} | " 
        f"time {epoch_time:.2f}s"
    )
training_end_time = time.perf_counter()
total_training_time = (
    training_end_time 
    - training_start_time
)

average_epoch_time = (
    sum(epoch_time_history)
    / len(epoch_time_history)
)
print()
print(
    f"Total training time: "
    f"{total_training_time:.2f}s"
)

print(
    f"Average epoch time: "
    f"{average_epoch_time:.2f}s"
)


# ==============================
# 1. Loss 曲线
# ==============================

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
plt.title('CNN Training and Test Loss')

plt.legend()
plt.grid()


# ==============================
# 2. Accuracy 曲线
# ==============================

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
plt.title('CNN Training and Test Accuracy')

plt.legend()
plt.grid()


# ==============================
# 3. 获取一个测试 batch 的预测结果
# ==============================

model.eval()

with torch.no_grad():

    test_images, test_labels = next(iter(test_loader))

    logits = model(test_images)

    predictions = torch.argmax(
        logits,
        dim=1
    )


# ==============================
# 4. 显示前 16 张图片及预测结果
# ==============================

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

plt.suptitle('CNN MNIST Predictions')

plt.tight_layout()


# ==============================
# 5. 显示预测错误的图片
# ==============================

wrong_indices = (
    predictions != test_labels
).nonzero(as_tuple=True)[0]


if wrong_indices.numel() > 0:

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

    plt.suptitle('CNN Wrong Predictions')

    plt.tight_layout()

else:

    print("No wrong predictions in this batch.")


plt.show()
