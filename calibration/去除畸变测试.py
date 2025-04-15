# -*- coding: utf-8 -*-
# @Time    : 2025/4/6 21:14
# @Author  : sjh
# @Site    : 
# @File    : 去除畸变测试.py
# @Comment :
import cv2
import numpy as np
fx = 229.98088073730466
fy = 229.98088073730466
cx = 329.4670867919922
cy = 206.48446655273438
# 假设 K 和 dist 是你从标定中获得的相机内参矩阵和畸变系数
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # 示例内参矩阵
dist = np.array([0.13352317, -0.40643628, -0.001092, 0.0012349, 0.30457482])  # 示例畸变系数

# 读取图像
image = cv2.imread("data/left/15.png")

# 去畸变
new_mtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (image.shape[1], image.shape[0]), 1, (image.shape[1], image.shape[0]))
undistorted_image = cv2.undistort(image, K, dist, None, new_mtx)

# 显示去畸变后的图像
cv2.imshow("Undistorted Image", undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
