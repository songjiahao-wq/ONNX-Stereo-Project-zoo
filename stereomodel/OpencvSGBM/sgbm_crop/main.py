# -*- coding: utf-8 -*-
# @Time    : 2025/5/23 22:16
# @Author  : sjh
# @Site    : 
# @File    : main.py
# @Comment :
import cv2
import numpy as np

from stereomodel.OpencvSGBM.utils.SGBM import SGBM
from config import Stereo
from stereomodel.OpencvSGBM.utils.filterz_disp import fill_depth_nearest_ROI
SGBM_ins = SGBM(use_blur=True)
Stereo_ins = Stereo()
if __name__ == '__main__':
    left_img = cv2.imread('left_img.png')
    right_img = cv2.imread('right_img.png')
    left_img = cv2.resize(left_img, (640, 480))
    right_img = cv2.resize(right_img, (640, 480))
    x1y1 = [218,103]
    x2y2 = [395, 458]
    left_img = left_img[x1y1[1]:x2y2[1], x1y1[0]:x2y2[0]]
    right_img = right_img[x1y1[1]:x2y2[1], x1y1[0]:x2y2[0]]
    disp = SGBM_ins.estimate_depth(left_img, right_img)
    # disp = cv2.imread('disp.png', cv2.IMREAD_UNCHANGED)
    print(disp.dtype)
    # disp = fill_depth_nearest_ROI(disp.astype(np.uint8))
    Stereo_ins.show_depth_point(disp, left_img)
