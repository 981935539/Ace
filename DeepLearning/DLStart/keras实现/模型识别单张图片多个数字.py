
import cv2
import numpy as np

from keras.datasets import mnist
from keras.models import load_model


## 数据预处理工具方法
def x_sort(contour):
    # 通过计算质心来实现从左到右排序
    M = cv2.moments(contour) # cx, 质心的x轴坐标
    return (int(M['m10']/M['m00']))


def make_square(not_square):
    # 画矩形包裹
    BLACK = [0,0,0]
    img_dim = not_square.shape
    height = img_dim[0]
    width = img_dim[1]
    #print("Height = ", height, "Width = ", width)
    if (height == width):
        square = not_square
        return square
    else:
        doublesize = cv2.resize(not_square,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
        height = height * 2
        width = width * 2
        #print("New Height = ", height, "New Width = ", width)
        if (height > width):
            pad = int((height - width)/2)
            #print("Padding = ", pad)
            doublesize_square = cv2.copyMakeBorder(doublesize, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=BLACK)
        else:
            pad = int((width - height)/2)
            #print("Padding = ", pad)
            doublesize_square = cv2.copyMakeBorder(doublesize, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
    doublesize_square_dim = doublesize_square.shape
    #print("Sq Height = ", doublesize_square_dim[0], "Sq Width = ", doublesize_square_dim[1])
    return doublesize_square


def resize_to_pixel(dimensions, image):
    # 缩放
    buffer_pix = 4
    dimensions  = dimensions - buffer_pix
    squared = image
    r = float(dimensions) / squared.shape[1]
    dim = (dimensions, int(squared.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    img_dim2 = resized.shape
    height_r = img_dim2[0]
    width_r = img_dim2[1]
    BLACK = [0,0,0]
    if (height_r > width_r):
        resized = cv2.copyMakeBorder(resized,0,0,0,1,cv2.BORDER_CONSTANT,value=BLACK)
    if (height_r < width_r):
        resized = cv2.copyMakeBorder(resized,1,0,0,0,cv2.BORDER_CONSTANT,value=BLACK)
    p = 2
    ReSizedImg = cv2.copyMakeBorder(resized,p,p,p,p,cv2.BORDER_CONSTANT,value=BLACK)
    img_dim = ReSizedImg.shape
    height = img_dim[0]
    width = img_dim[1]
    #print("Padded Height = ", height, "Width = ", width)
    return ReSizedImg
# model.predict(np.array([[0,1]]))


image = cv2.imread('images/aaa.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("image", image)
cv2.waitKey(0)

# 高斯模糊
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# cv2.imshow("blurred", blurred)
# cv2.waitKey(0)

edged = cv2.Canny(blurred, 30, 150)
# cv2.imshow("edged", edged)
# cv2.waitKey(0)

# 查找边缘
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 从左到右排序坐标
contours = sorted(contours, key=x_sort, reverse=False)

# 存储数据的数组
full_number = []
classifier = load_model('/home/yang-pc/JupyterNotebook/DeepLearning/day05/mnist_simple_cnn.h5')

# 循环兴趣点
for c in contours:
    # 矩形边框包裹感兴趣的区域
    (x, y, w, h) = cv2.boundingRect(c)

    if w >= 5 and h >= 25:
        roi = blurred[y:y + h, x:x + w]
        ret, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
        roi = make_square(roi)  # 转换成正方形
        roi = resize_to_pixel(28, roi)  # 缩放到28*28
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)
        roi = roi / 255.0
        roi = roi.reshape(1, 28, 28, 1)

        ## 预测
        res = str(classifier.predict_classes(roi, 1, verbose=0)[0])  # 返回类别索引
        full_number.append(res)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, res, (x, y + 155), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
        cv2.imshow("image", image)
        cv2.waitKey(0)

# cv2.waitKey(0)
cv2.destroyAllWindows()
print("图片中的数字为: " + ''.join(full_number))