import keras
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import load_model
import matplotlib.pyplot as plt

# 加载数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# 归一化
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# 数据归一化 (0 to 255) -> (0 to 1)
x_train /= 255
x_test /= 255

# one-hot
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 加载模型
model = load_model("mnist_simple_cnn.h5")




# 获取模型的每一层，卷积层，全连接层，激活层，池化层，BatchNormalization
layers = model.layers
# 获取卷积层的名称和输出变量
layers_names = []
layers_output = []
for layer in layers:
    if 'conv2d' in layer.output.name:
        layers_names.append(layer.output.name)
        layers_output.append(layer.output)


# 分层获取模型输出
layers_model = keras.models.Model(inputs=model.input, outputs=layers_output)
outputs = layers_model.predict(x_test[1].reshape(1,28,28,1))

# 按卷积层可视化
images_per_row = 16
for layer_name, output in zip(layers_names, outputs):
    n_features = output.shape[-1]
    size = output.shape[1]

    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = output[0, :, :, col * images_per_row + row]

            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size,
            row * size: (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='gray')

plt.show()