# coding:utf-8

import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.metrics import classification_report,confusion_matrix

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


# 预测测试数据
y_pred = model.predict_classes(x_test, verbose=1)
y_true = np.argmax(y_test, axis=1)

# 预测报告
report = classification_report(y_true, y_pred)
print(report)

# 混淆矩阵
matrix = confusion_matrix(y_true, y_pred)
print(matrix)


# 所有预测错误的下标
result = np.absolute(y_true - y_pred)
#只要真实值和预测值不一样就获取位置
result_index = np.nonzero(result > 0)
print(result_index)


def draw_test(name, pred, input_im, true_label):
    BLACK = [0, 0, 0]
    expanded_image = cv2.copyMakeBorder(input_im, 0, 0, 0, imageL.shape[0] * 2, cv2.BORDER_CONSTANT, value=BLACK)
    expanded_image = cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)
    cv2.putText(expanded_image, str(pred), (152, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (0, 255, 0), 2)
    cv2.putText(expanded_image, str(true_label), (250, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (0, 0, 255), 2)
    cv2.imshow(name, expanded_image)


for i in range(0, 10):
    input_im = x_test[result_index[0][i]]
    imageL = cv2.resize(input_im, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    input_im = input_im.reshape(1, 28, 28, 1)

    ## Get Prediction
    res = str(model.predict_classes(input_im, 1, verbose=0)[0])
    draw_test("Prediction", res, imageL, np.argmax(y_test, axis=1)[result_index[0][i]])
    cv2.waitKey(0)

cv2.destroyAllWindows()