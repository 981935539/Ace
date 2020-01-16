# encoding=utf-8

import cv2
import numpy as np
from keras.models import load_model


## 数据预处理工具方法
def x_sort(contour):
    # 通过计算质心来实现从左到右排序
    M = cv2.moments(contour) # cx, 质心的x轴坐标
    return (int(M['m10']/(M['m00']+1e-4)))


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

if __name__ == '__main__':
    # 获取摄像头
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("相机无法打开！请检查重试")
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)


    classifier = load_model('mnist_simple_cnn.h5')

    while True:
        flag, frame = video.read()
        # 获取图片成功
        if not flag:
            print("未读到图片")
            break

        cv2.imshow("frame", frame)
        key = cv2.waitKey(40)

        # # 灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # 高斯滤波
        gauss = cv2.GaussianBlur(gray, (3, 3), 1.0)
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # 限制对比度的自适应阈值均衡化
        dst = clahe.apply(gauss)
        # 自适应
        adapt_img = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # 中值滤波,去除椒盐噪声
        median = cv2.medianBlur(adapt_img, 5)
        # 形态学去除噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closing = cv2.morphologyEx(median, cv2.MORPH_CLOSE, kernel, iterations=1)
        edged = cv2.Canny(closing, 30, 150)

        # 查找边缘
        contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print len(contours)

        # 从左到右排序坐标
        contours = sorted(contours, key=x_sort, reverse=False)

        # 存储数据的数组
        full_number = []
        # 循环兴趣点
        for c in contours:
            # 矩形边框包裹感兴趣的区域
            (x, y, w, h) = cv2.boundingRect(c)

            if w >= 5 and h >= 25:
                roi = closing[y:y + h, x:x + w]
                ret, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
                roi = make_square(roi)  # 转换成正方形
                roi = resize_to_pixel(28, roi)  # 缩放到28*28
                cv2.imshow("ROI", roi)
                # cv2.waitKey(0)
                roi = roi / 255.0
                roi = roi.reshape(1, 28, 28, 1)

                ## 预测
                res = str(classifier.predict_classes(roi, 1, verbose=0)[0])  # 返回类别索引

                # full_number.append(res)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, res, (x, y + 155), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
                cv2.imshow("pymat", frame)

        # esc退出
        if key == 27:
            break