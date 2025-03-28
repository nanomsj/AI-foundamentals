2400010833 闵诗珈

# 人工智能基础

### 作业二

### part 1 用numpy实现训练MLP网络识别手写数字MNIST数据集

1. 补充代码内容并进行训练:
   - 使用交叉熵损失函数进行训练

2. 调整激活函数

   - sigmoid

     epoch: 10, val_loss: 0.2490, val_accuracy: 0.9297

   - tanh

     epoch: 10, val_loss: 0.1509, val_accuracy: 0.9580

     **结论**：训练效果：RELU>tanh>标准（94%）>sigmoid

3. 更改损失函数：使用hinge损失函数（因为本来用的就是提示的交叉熵损失函数，而均方差一般不用于这里的多分类问题，所以这里尝试更换误差函数的代码是从网上搜索得到的）

   - **Hinge 损失函数**强调正确分类的样本具有足够大的间隔，常用于支持向量机等分类模型，在处理线性可分或近似线性可分的数据时表现较好。

   epoch: 10, val_loss: 0.4478, val_accuracy: 0.9556

3. 调整网格结构：使用三层的神经网络

epoch: 10, val_loss: 0.1434, val_accuracy: 0.9707

**结论**：训练成本（时间）增加，效果无显著差异。



### part 2 使用Pytorch训练MNIST数据集的MLP模型

1. 由于笔记本没有NVIDIA显卡，改为使用cpu进行训练，训练速度较慢

2. 使用SGD优化器

Validation set: Epoch 10 - Average loss: 0.1990, Accuracy: 94.23%

**结论**：准确率未达标

3. 改为使用Adam优化器

   并且添加Dropout层以减少过拟合

运行结果：

Validation set: Average loss: 0.0679, Accuracy: 9816/10000 (98.16%)

**结论**：同时使用以上两种优化方法可以显著提高准确性

4. 增加全连接层和层内神经元数量

Validation set: Average loss: 0.0839, Accuracy: 9788/10000 (97.88%)

**结论**：减慢训练速度，对准确度无显著影响



### **代码**

###### part 1

```python
# -*- coding: utf-8 -*-
"""
@ author: 闵诗珈
"""

# 作业内容：更改loss函数、网络结构、激活函数，完成训练MLP网络识别手写数字MNIST数据集

import numpy as np
from tqdm import tqdm

#import os
#print(os.getcwd())

# 加载数据集,numpy格式
X_train = np.load('./mnist/X_train.npy')  # (60000, 784), 数值在0.0~1.0之间
y_train = np.load('./mnist/y_train.npy')  # (60000, )
y_train = np.eye(10)[y_train]  # (60000, 10), one-hot编码

X_val = np.load('./mnist/X_val.npy')  # (10000, 784), 数值在0.0~1.0之间
y_val = np.load('./mnist/y_val.npy')  # (10000,)
y_val = np.eye(10)[y_val]  # (10000, 10), one-hot编码

X_test = np.load('./mnist/X_test.npy')  # (10000, 784), 数值在0.0~1.0之间
y_test = np.load('./mnist/y_test.npy')  # (10000,)
y_test = np.eye(10)[y_test] # (10000, 10), one-hot编码

# 定义激活函数
def relu(x):
    '''
    relu函数，i.e.f(x)=max(0,x)
    '''
    return np.maximum(0, x)

def relu_prime(x):
    '''
    relu函数的导数
    '''
    # return 1 if x>0 else 0
    # 使用np包如下：
    return np.where(x > 0, 1, 0)

# sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid函数的导数
def sigmoid_prime(x):
    sigmoid(x)
    return sigmoid(s) * (1 - sigmoid(s))

# tanh函数
def tanh(x):
    return np.tanh(x)

# tanh函数的导数
def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

#输出层激活函数
def softmax(x):
    '''
    softmax函数, 防止除0
    '''
    x_stab = np.exp(x - np.max(x, axis=1, keepdims=True)) # 减去最大值，避免数据溢出
    return x_stab / np.sum(x_stab, axis=1, keepdims=True)

def softmax_prime(x):
    '''
    softmax函数的导数
    '''
    return softmax(x) * (1.0-softmax(x))

# 定义损失函数 交叉熵
def loss_fn(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出，防0
    '''
    return -np.sum(y_true * np.log(y_pred + 1e-8), axis=-1)

def loss_fn_prime(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    return y_pred - y_true

# 定义Hinge损失函数
'''
def loss_fn(y_true, y_pred):
  
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出，防0
 
    margin = 1.0
    correct_class_scores = np.sum(y_true * y_pred, axis=1, keepdims=True)
    margins = np.maximum(0, y_pred - correct_class_scores + margin)
    margins[y_true.astype(bool)] = 0
    return np.mean(np.sum(margins, axis=1))

def loss_fn_prime(y_true, y_pred):
    
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    
    margin = 1.0
    correct_class_scores = np.sum(y_true * y_pred, axis=1, keepdims=True)
    margins = np.maximum(0, y_pred - correct_class_scores + margin)
    binary_margins = (margins > 0).astype(float)
    binary_margins[y_true.astype(bool)] = -np.sum(binary_margins, axis=1)
    return binary_margins / y_true.shape[0]
    '''


# 定义权重初始化函数
def init_weights(shape):
    return np.random.normal(loc=0.0, scale=np.sqrt(2.0 / shape[0]), size=shape)

# 定义网络结构
class Network(object):
    '''
    MNIST数据集分类网络
    '''
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        '''
        初始化网络结构
        两层全连接神经网络
        input_size=784, hidden_size=256, output_size=10, lr=0.01
        '''
        self.W1 = init_weights((input_size, hidden_size)) # 输入层到隐藏层的权重矩阵
        self.b1 = np.zeros(hidden_size)  # 输入层到隐藏层的偏置
        self.W2 = init_weights((hidden_size, output_size)) # 隐藏层到输出层的权重矩阵
        self.b2 = np.zeros(output_size)  # 隐藏层到输出层的偏置
        self.lr = lr  # 学习率

    def forward(self, x):
        '''
        前向传播
        '''
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = softmax(self.z2)
        return self.a2

    def step(self, x_batch, y_batch):
        '''
        一步训练
        '''

        # 前向传播
        y_pred = self.forward(x_batch)

        # 计算损失和准确率
        loss = np.mean(loss_fn(y_batch, y_pred))
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))

        # 反向传播
        dz2 = loss_fn_prime(y_batch, y_pred)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0)

        dz1 = np.dot(dz2, self.W2.T) * relu_prime(self.a1)
        dW1 = np.dot(x_batch.T, dz1)
        db1 = np.sum(dz1, axis=0)

        # 更新权重
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        return loss, accuracy

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

if __name__ == '__main__':
    # 训练网络
    net = Network(input_size=784, hidden_size=256, output_size=10, lr=0.01)
    for epoch in range(10):
        losses = []
        accuracies = []

        p_bar = tqdm(range(0, len(X_train), 64))
        for i in p_bar:
            x_batch = X_train[i:i + 64]
            y_batch = y_train[i:i + 64]
            loss, accuracy = net.step(x_batch, y_batch)

            losses.append(loss)
            accuracies.append(accuracy)

        # 验证网络
        y_pred=net.forward(X_val)
        val_loss=np.mean(loss_fn(y_val, y_pred))
        val_accurary=np.mean(np.argmax(y_pred, axis=-1) == np.argmax(y_val, axis=-1))
        print(f"epoch: {epoch + 1}, val_loss: {val_loss:.4f}, val_accuracy: {val_accurary:.4f}")
```

###### part 2

```python
# 第一课作业
# 使用Pytorch训练MNIST数据集的MLP模型
# 1. 运行、阅读并理解mnist_mlp_template.py,修改网络结构和参数，增加隐藏层，观察训练效果
# 2. 使用Adam等不同优化器，添加Dropout层，观察训练效果
# 要求：10个epoch后测试集准确率达到97%以上

# 导入相关的包
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

# 加载数据集,numpy格式
X_train = np.load('./mnist/X_train.npy')
y_train = np.load('./mnist/y_train.npy')
X_val = np.load('./mnist/X_val.npy')
y_val = np.load('./mnist/y_val.npy')
X_test = np.load('./mnist/X_test.npy')
y_test = np.load('./mnist/y_test.npy')

# 定义MNIST数据集类
class MNISTDataset(Dataset):  # 继承Dataset类

    def __init__(self, data=X_train, label=y_train):
        '''
        Args:
            data: numpy array, shape=(N, 784)
            label: numpy array, shape=(N, 10)
        '''
        self.data = data
        self.label = label

    def __getitem__(self, index):
        '''
        根据索引获取数据,返回数据和标签,一个tuple
        '''
        data = self.data[index].astype('float32')  # 转换数据类型, 神经网络一般使用float32作为输入的数据类型
        label = self.label[index].astype('int64')  # 转换数据类型, 分类任务神经网络一般使用int64作为标签的数据类型
        return data, label

    def __len__(self):
        '''
        返回数据集的样本数量
        '''
        return len(self.data)

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 800)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(800, 800)
        self.dropout2 = nn.Dropout()
        self.fc3 = nn.Linear(800, 10)

    def forward(self, x):
        x = x.view(-1, 784)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

# 实例化模型
model = Net()
model.to(device='cpu')
# 定义损失函数
criterion = nn.CrossEntropyLoss()
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8)

# 定义数据加载器
train_loader = DataLoader(MNISTDataset(X_train, y_train),
                          batch_size=64, shuffle=True)
val_loader = DataLoader(MNISTDataset(X_val, y_val),
                        batch_size=64, shuffle=True)
test_loader = DataLoader(MNISTDataset(X_test, y_test),
                         batch_size=64, shuffle=True)

# 定义训练参数
EPOCHS = 10

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
# 训练模型
for epoch in range(EPOCHS):
    # 训练模式
    model.train()

    loss_train = []
    acc_train = []
    correct_train = 0


    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1} Training')):
        data, target = data.to(device='cpu'), target.to(device='cpu')
        # 梯度清零
        optimizer.zero_grad()
        # 前向计算
        output = model(data)
        # 计算损失
        loss = criterion(output, target)
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
        # 打印训练信息
        '''if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))''' 
        
        loss_train.append(loss.item())
        pred = output.max(1, keepdim=True)[1]  # 获得最大对数概率的索引

        correct = pred.eq(target.view_as(pred)).sum().item()
        correct_train += correct
        acc_train.append(100. * correct / len(data))

    epoch_train_loss = np.mean(loss_train)
    epoch_train_acc = np.mean(acc_train)
    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc)
    print(f'Train set: Epoch {epoch + 1} - Average loss: {epoch_train_loss:.4f}, Accuracy: {100. * correct_train / len(train_loader.dataset):.2f}%')

    # 测试模式
    model.eval()
    val_loss = []
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device='cpu'), target.to(device='cpu')
            
            output = model(data)
            val_loss.append(criterion(output, target).item()) # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss = np.mean(val_loss)

    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

    history['val_loss'].append(val_loss)
    history['val_acc'].append(100. *correct / len(val_loader.dataset))

# 画图
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='train_loss')
plt.plot(history['val_loss'], label='val_loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='train_acc')
plt.plot(history['val_acc'], label='val_acc')
plt.legend()
plt.show()
```

