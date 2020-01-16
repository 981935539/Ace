# 加载图片
import cv2
from keras.models import load_model


# 加载模型
# model = load_model("mnist_simple_cnn.h5")
model = load_model("mnist_simple_cnn_9.h5")
# model.summary()

# 读取图片
image = cv2.imread("images/test3.png")
# cv2.imshow("image", image)
# cv2.waitKey(0)

# 灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 高斯模糊
blured = cv2.GaussianBlur(gray, (5, 5), 0)


# Candy边缘提取
edged = cv2.Canny(blured, 30, 150)


# 提取轮廓
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(contours)

for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    #     if w >= 5 and h >= 25:
    # 提取数字
    roi = cv2.resize(gray[y - 20:y + h + 20, x - 20:x + w + 20], (28, 28))
    roi = 255 - roi  # 转换为黑色背景
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)
    # 处理图片
    roi = roi / 255.0
    roi = roi.reshape(1, 28, 28, 1)

    # 预测图片
    res = model.predict_classes(roi, 1, verbose=0)[0]
    print(res)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0))
    cv2.putText(image, str(res), (x + 10, y + 200), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 2)
    cv2.imshow("image", image)

    cv2.waitKey(0)

cv2.destroyAllWindows()