import cv2
import numpy as np
from stereomodel.OpencvSGBM.utils.SGBM import SGBM

# 初始化SGBM
stereo = SGBM(use_blur=True).sgbm

# 加载左右图像（这里仍然用灰度图，但你可以尝试彩色图）
left_img = cv2.imread('../../data/mid/im0.png', cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread('../../data/mid/im1.png', cv2.IMREAD_GRAYSCALE)

# 计算左图视差图（左视图作为参考）
disparity_left = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

# 计算右图视差图（右视图作为参考）
right_matcher = cv2.StereoSGBM_create(
    minDisparity=-stereo.getNumDisparities(),
    numDisparities=stereo.getNumDisparities(),
    blockSize=stereo.getBlockSize(),
    P1=stereo.getP1(),
    P2=stereo.getP2(),
    disp12MaxDiff=stereo.getDisp12MaxDiff(),
    uniquenessRatio=stereo.getUniquenessRatio(),
    speckleWindowSize=stereo.getSpeckleWindowSize(),
    speckleRange=stereo.getSpeckleRange(),
    mode=stereo.getMode()
)
disparity_right = right_matcher.compute(right_img, left_img).astype(np.float32) / 16.0

# 左右一致性检查（LRC）
lrc_threshold = 10  # 可调整
height, width = disparity_left.shape
mask_consistent = np.ones((height, width), dtype=np.uint8) * 255  # 初始化有效区域

for y in range(height):
    for x in range(width):
        right_x = int(x - disparity_left[y, x])  # 左视差对应的右图像素坐标
        # 检查 right_x 是否在有效范围内
        if 0 <= right_x < width:
            if abs(disparity_left[y, x] - disparity_right[y, right_x]) > lrc_threshold:
                mask_consistent[y, x] = 0  # 不一致，标记为无效
        else:
            mask_consistent[y, x] = 0  # 越界，标记为无效

# 应用一致性掩码，剔除不一致的视差
disparity_left_filtered = disparity_left.copy()
disparity_left_filtered[mask_consistent == 0] = 0  # 设为0或插值填充

# 可选：对无效区域进行插值（如最近邻或中值滤波）
# disparity_left_filtered = cv2.inpaint(disparity_left_filtered, (mask_consistent == 0).astype(np.uint8), 3, cv2.INPAINT_TELEA)

# 标准化视差图并转换为颜色图
norm = ((disparity_left_filtered - disparity_left_filtered.min()) /
        (disparity_left_filtered.max() - disparity_left_filtered.min()) * 255).astype(np.uint8)
depth_colormap1 = cv2.applyColorMap(norm, cv2.COLORMAP_PLASMA)

# 显示结果
cv2.imshow('Original Disparity', ((disparity_left - disparity_left.min()) / (disparity_left.max() - disparity_left.min()) * 255).astype(np.uint8))
cv2.imshow('Filtered Disparity (LRC)', depth_colormap1)
cv2.waitKey(0)
cv2.destroyAllWindows()