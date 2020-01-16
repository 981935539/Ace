# 加载图片
import cv2
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


# ++++++++++++++++++++++++++++++++++++++创建要增强的数据
# 读取图片
image = cv2.imread("images/test5.png")
# 灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 高斯模糊
blured = cv2.GaussianBlur(gray, (5,5), 0)
# Candy边缘提取
edged = cv2.Canny(blured, 30, 150)
# 提取轮廓
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
#     if w >= 5 and h >= 25:
    # 提取数字
    roi = cv2.resize(gray[y-20:y+h+20, x-20:x+w+20],(28,28))
    cv2.imwrite("images/9.jpeg", roi)


# +++++++++++++++++++++++++++++++生成增强数据+++++++++++++++++++++++++++++++

data_gen = ImageDataGenerator(rotation_range=20,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              shear_range=0.2, #剪切
                              zoom_range=0.2,
                              horizontal_flip=False, # 镜像
                              fill_mode="nearest")

img = load_img("images/9.jpeg")
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in data_gen.flow(x, save_to_dir='output', save_prefix='9', save_format='jpeg'):
    i += 1
    if i>39:
        break



# ++++++++++++++++++++++++ 处理重训练数据+++++++++++
# 处理数据

features = []
files = os.listdir('./output')
for file in files:
    path = os.path.join('./output', file)
    data = cv2.imread(path)
    gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    gray = (255 - gray)
    features.append(gray.reshape(gray.shape[0], gray.shape[1], -1))

features = np.array(features) / 255.0

train_size = features.shape[0]

labels = np.eye(10, dtype=np.int)[np.ones(train_size, dtype=np.int) * 9]

# +++++++++++++++++++++++++++++++++++=微调模型
# 微调模型
# 加载模型
model = load_model("mnist_simple_cnn.h5")
model.fit(features, labels, batch_size=10, epochs=10)

# +++++++++++++++++++++++++++++++++++++保存模型
model.save("mnist_simple_cnn_9.h5")

