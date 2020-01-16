import keras
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers import Dense, Softmax, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, Activation


def load_data():
    """
    加载数据集
    :return: (x_train, y_train), (x_test, y_test)
    """
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

    return (x_train, y_train), (x_test, y_test)


def build_model():
    """
    4层卷积层，2层全连接层
    • 基于3× 3的小型滤波器的卷积层。
    • 激活函数是ReLU。
    • 全连接层的后面使用Dropout层。
    • 基于Adam的最优化。
    • 使用He初始值作为权重初始值。
    accuracy: 0.996299
    :return:
    """
    model = Sequential()
    # , kernel_initializer='he_normal'
    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3) , kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, kernel_initializer='he_normal') )
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    # 加载数据
    (x_train, y_train), (x_test, y_test) = load_data()
    # 构建模型
    model = build_model()

    # 训练模型
    epochs = 20
    batch_size = 128
    hist = model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_test, y_test), epochs=epochs, verbose=0)

    # 测试模型
    scores = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("test loss", scores[0])
    print("test accuracy", scores[1])

    # 保存模型
    model.save("mnist_simple_cnn.h5")