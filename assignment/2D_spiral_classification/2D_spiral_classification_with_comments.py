import numpy as np
import matplotlib.pyplot as plt

#超参数
num_per_class = 100
num_classes = 3
noise = 0.2
seed = 0
num_steps = 5000
input_dim = 2
hidden_dim = 100
output_dim = 3 
learning_rate = 1

#二维螺旋数据生成器
def make_2D_spiral_data (num_per_class,num_classes,noise,seed):

    num_samples = num_per_class * num_classes
    X = np.zeros((2,num_samples))
    y = np.zeros(num_samples,dtype=int)

    rg = np.random.default_rng(seed)

    for j in range(num_classes):

        idx = slice(j*num_per_class,(j+1)*num_per_class)

        r = np.linspace(0,1,num_per_class)

        theta = np.linspace(j*4,(j+1)*4,num_per_class)

        theta = theta + rg.normal(0,noise,num_per_class)
        #theta = theta + rg.random(num_per_class)

        X[0,idx] = r * np.sin(theta)
        X[1,idx] = r * np.cos(theta)

        y[idx] = j

    return X,y

X,y = make_2D_spiral_data(num_per_class,num_classes,noise,seed) 

#One-hot 标签
num_samples = num_per_class * num_classes
Y = np.zeros((num_classes,num_samples)) 
#花式索引 y.shape=(300,) np.arange(num_samples).shape=(300,) 前面的当行坐标 后面的当列坐标
Y[y,np.arange(num_samples)] = 1 

#画图
plt.scatter(X[0,:],X[1,:],c=y)
#plt.show()

#X.shape = (2,300)
#y.shape = (300,)
#Y.shape = (3,300)

#激活函数 ReLU
def relu(z):
    return np.maximum(z,0)

#softmax
def softmax(z):

    z = z - np.max(z,axis = 0,keepdims = True) 
    #axis把行删除掉 实际上就是求每一列的最大值,keepdims = True,把行变成1
    #如果没有keepdims = True,np.max(z,axis=0).shape = (300,)是这300列每一列的最大值
    #加keepdims = True之后，np.max(z,axis = 0,keepdims = True).shape = (1,300)删除的行变成1, 
    #ChatGPT:z.shape = (3,300)不加keepdims = True其实也可以广播,不过为了让维度含义更清楚以及在复杂代码中减少出错,加上keepdims=1
    
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z,axis=0,keepdims=1) 
    #同样的道理 我们要把每一列的p求和,所以axis=0,之后在把行设为1,进行除法广播

    #初始化参数
rg = np.random.default_rng(seed)

w1 = rg.standard_normal((hidden_dim,input_dim)) * 0.1 #(100,2)
b1 = np.zeros((hidden_dim,1))           #(100,1) 

w2 = rg.standard_normal((output_dim,hidden_dim)) * 0.1#(3,100)
b2 = np.zeros((output_dim,1))           #(3,1)

for step in range(num_steps):
    #前向传播
    #隐藏层
    z1 = w1 @ X + b1 
    a1 = relu(z1)
    #输出层
    z2 = w2 @ a1 + b2

    probs = softmax(z2) #probs.shape = (3,300)

    #在这个例子中 由于使用one-hot标签,非真实类别的对应的标签为0，所以我们只取真实类别的概率做计算也可以 而且也不用乘标签了 因为真实类别的标签为1
    #取真实类别的概率 花式索引
    correct_probs = probs[y,np.arange(num_samples)] #correct_probs.shape = (300,)
    loss1 = -np.mean(np.log(correct_probs + 1e-12))
    #在numpy的实现中loss就是写来看的 实际的影响反向传播的是我们写的delta2 = ... 二者需要对应
    #这是完整的交叉熵损失写法
    loss2 = - np.mean(np.sum(Y*np.log(probs+1e-12),axis = 0))
    #反向传播
    delta2 = (probs - Y)/num_samples 
    #delta2.shape = (3,300)关于 loss --> z2 的求导过程 这里不多叙述 
    #关于为什么需要除以300,原因使我们后面所有的梯度计算都对应于平均loss???
    #后面的d不需要再除以num_samples是因为delta2已经平均过了
    dw2 = delta2 @ a1.T #(3,300) @ (300,100) = (3,100)
    db2 = np.sum(delta2,axis=1,keepdims=True) #(3,1) 

    delta1 = (w2.T @ delta2) * (z1 > 0) #(100,3) @ (3,300) = (100,300)
    dw1 = delta1 @ X.T                  #(100,300) @ (300,2) = (100,2)
    db1 = np.sum(delta1,axis=1,keepdims=True) #(100,1)

    #梯度下降
    w1 -= dw1 * learning_rate
    b1 -= db1 * learning_rate

    w2 -= dw2 * learning_rate
    b2 -= db2 * learning_rate

    #打印进度
    if step % 500 == 0:
        prediction = np.argmax(probs,axis=0) #(300,)
        accuracy = np.mean(y == prediction)

        print(f"step:{step:4d}")
        print(f"loss:{loss1:.4f}")
        print(f"accuracy:{accuracy:.4f}")
        print(f"delta_loss{(loss1-loss2):.4f}")

#训练结束后 最后进行一次前向传播算出最终的准确率
z1 = w1 @ X + b1 
a1 = relu(z1)
z2 = w2 @ a1 + b2
probs = softmax(z2)
prediction = np.argmax(probs,axis=0)
accuracy = np.mean(y == prediction)
print(f"Final accuracy:{accuracy:.4f}")


