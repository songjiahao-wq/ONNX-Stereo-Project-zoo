# -*- coding: utf-8 -*-
# @Time    : 2025/6/8 19:55
# @Author  : sjh
# @Site    : 
# @File    : filterz_disp.py
# @Comment :
# -*- coding: utf-8 -*-
# @Time    : 2025/5/21 上午9:40
# @Author  : sjh
# @Site    :
# @File    : Filter.py
# @Comment :
import time

import cv2
import numpy as np
# cv2.setNumThreads(4)
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
def fill_depth_inpaint(depth_frame, left=100, right=10, bottom=20):
    # 1. 创建一个初始掩膜，值为0的像素为True，否则为False
    mask0 = (depth_frame == 0)

    height, width = depth_frame.shape
    mask1 = np.zeros_like(depth_frame, dtype=bool)
    mask1[0:height-bottom, left:width-right] = True
    mask = mask0 & mask1
    mask = np.uint8(mask * 255)
    # 使用 cv2.inpaint() 进行修复
    inpaint_radius = 3 # 修复半径，可以根据需要调整
    depth_frame = cv2.inpaint(depth_frame, mask, inpaint_radius, cv2.INPAINT_TELEA)

    return depth_frame

def fill_depth_nearest(depth_frame, left=200, right=0, bottom=0):
    # 创建一个掩码，表示需要修复的位置（为0 且在感兴趣区域内）
    mask0 = (depth_frame == 0)

    height, width = depth_frame.shape
    mask1 = np.zeros_like(depth_frame, dtype=bool)
    mask1[0:height-bottom, left:width-right] = True

    mask = mask0 & mask1  # 最终掩码：只修复感兴趣区域内的0值
    filled = depth_frame.copy()

    # 最近点填充
    distance, indices = distance_transform_edt(mask, return_indices=True)
    filled[mask] = depth_frame[tuple(indices[:, mask])]
    return filled

def fill_depth_nearest_ROI(depth_frame, left=64, right=0, bottom=0):
    # 深度图尺寸
    height, width = depth_frame.shape

    # 定义 ROI 区域（感兴趣区域）
    roi_top = 0
    roi_bottom = height - bottom
    roi_left = left
    roi_right = width - right

    # 截取 ROI 区域
    roi = depth_frame[roi_top:roi_bottom, roi_left:roi_right].copy()

    # 掩码：0 为需要填补的区域
    mask = roi == 0

    # 若无0值，直接返回原图
    if not np.any(mask):
        return depth_frame
    # 最近邻填充
    distance, indices = distance_transform_edt(mask, return_indices=True)
    roi_filled = roi.copy()
    roi_filled[mask] = roi[tuple(indices[:, mask])]

    # 将修复后的 ROI 放回原图
    filled_frame = depth_frame.copy()
    filled_frame[roi_top:roi_bottom, roi_left:roi_right] = roi_filled

    return filled_frame
def fast_kdtree_fill(depth):
    yx = np.argwhere(depth != 0)
    values = depth[depth != 0]
    tree = cKDTree(yx)

    yx_nan = np.argwhere(depth == 0)
    _, idx = tree.query(yx_nan)
    filled = depth.copy()
    filled[depth == 0] = values[idx]
    return filled
def fill_depth_distanceTransform(depth):
    filled = cv2.distanceTransform(depth, cv2.DIST_L2, 5)
    return filled
def show_disp(disp):
    norm = ((disp - disp.min()) / (disp.max() - disp.min()) * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(norm, cv2.COLORMAP_PLASMA)
    return depth_colormap
if __name__ == "__main__":

    # 读取深度图（16位）
    depth_map = cv2.imread('../disp.png', cv2.IMREAD_UNCHANGED)

    print(depth_map.shape, depth_map.dtype)

    inpainted_depth = fill_depth_nearest_ROI(depth_map)


    # 归一化到 0~255，用于可视化（避免16位深度显示不正确）
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)

    inpainted_norm = cv2.normalize(inpainted_depth, None, 0, 255, cv2.NORM_MINMAX)
    inpainted_norm = inpainted_norm.astype(np.uint8)


    depth_norm = show_disp(depth_norm)
    inpainted_norm = show_disp(inpainted_norm)
    # 拼接显示
    combined_img = np.hstack((depth_norm, inpainted_norm))
    cv2.imshow('Original vs Inpainted Depth Map (Normalized)', combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


