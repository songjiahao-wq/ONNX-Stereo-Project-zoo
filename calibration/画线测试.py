# -*- coding: utf-8 -*-
# @Time    : 2025/4/13 21:28
# @Author  : sjh
# @Site    : 
# @File    : 画线测试.py
# @Comment :
import cv2
import numpy as np

# 读取图像
rectified_left = cv2.imread('../data/640x352/im0.png')
rectified_right = cv2.imread('../data/640x352/im1.png')
for i in range(10):
    cv2.line(rectified_left, (0, 100 + i * 50), (1056, 100 + i * 50), (0, 0, 255), 1)
    cv2.line(rectified_right, (0, 100 + i * 50), (1056, 100 + i * 50), (0, 0, 255), 1)
combined_img = np.hstack((rectified_left, rectified_right))
cv2.imshow('combined_img', combined_img)
cv2.waitKey(0)