

# *******************************模型复杂度图表epochs-loss****************************************

import matplotlib.pyplot as plt

# keras训练记录
# 模型复杂度图表
history_dict = hist.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epoches = range(1, epochs+1)

fig = plt.figure(figsize=(8, 6))

line1 = plt.plot(epoches, val_loss_values, label='Validation Loss')
line2 = plt.plot(epoches, loss_values, label='Training Loss')

plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker = '*', markersize=10.0)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

# *******************************等高线****************************************
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib notebook
np.set_printoptions(threshold=np.inf)
def f(x, y):
    return x**2 / 20 +  y**2

x = np.arange(-10, 10, 0.01)
y = np.arange(-5, 5, 0.01)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

Z[Z>7] = 0

x = np.arange(-10, 10, 0.01)
y = np.arange(-5, 5, 0.01)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

Z[Z>7] = 0


plt.contour(X, Y, Z)
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-10, 10)
plt.xlim(-10, 10)
plt.plot(0, 0, '+')
plt.show()

# *******************************梯度图****************************************
def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值

    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad


def function_2(x):
    if x.ndim == 1:
        return np.sum(x ** 2)
    else:
        return np.sum(x ** 2, axis=1)


def tangent_line(f, x):
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d * x
    return lambda t: d * t + y


x0 = np.arange(-2, 2.5, 0.25)
x1 = np.arange(-2, 2.5, 0.25)
X, Y = np.meshgrid(x0, x1)

X = X.flatten()
Y = Y.flatten()

grad = numerical_gradient(function_2, np.array([X, Y]))

plt.figure()
# 　X, Y　箭头的位置
# U, V, 箭头在X, Y轴的向量
# angles="xy", ‘xy’: arrows point from (x,y) to (x+u, y+v). 箭头角度方向
plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="red")  # ,headwidth=10,scale=40,color="#444444")
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('x0')
plt.ylabel('x1')
plt.grid()
plt.draw()
plt.show()



# ********************************************************
fig, ax = plt.subplots(nrows=10, ncols=10, sharex='all', sharey='all')
ax = ax.flatten()
for i in range(100):
    img = x_train[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

