# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from ANN.dataset.mnist import load_mnist
from ANN.common.multi_layer_net import MultiLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
network = MultiLayerNet(input_size=784, hidden_size_list=[100,],
                                  output_size=10, weight_init_std=0.01)
for key, val in network.params.items():
    network.params[key] = np.zeros_like(network.params[key])
print(network.params)

# iters_num = 10000  # 适当设定循环的次数 10000
iters_num = 2000
train_size = x_train.shape[0]
batch_size = 256
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 每个epoch迭代的次数
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # if i % iter_per_epoch == 0:
    if i % 100 == 0:
        # train_acc = network.accuracy(x_train, t_train)
        # test_acc = network.accuracy(x_test, t_test)
        # train_acc_list.append(train_acc)
        # test_acc_list.append(test_acc)
        print("train_loss | " + str(loss))

# 绘制图形
x = np.arange(iters_num)
plt.plot(x, train_loss_list, label='train loss')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()