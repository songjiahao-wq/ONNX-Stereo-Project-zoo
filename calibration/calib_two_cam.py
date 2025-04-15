import os
import numpy as np
import cv2
import glob

# 左右相机的内参和畸变系数
K_left = np.array([[745.36569178, 0., 540.56511955],
                   [0., 746.73646007, 386.53044259],
                   [0., 0., 1.]])  # 左相机内参

K_right = np.array([[743.30812943, 0., 541.17968337],
                    [0., 744.44363451, 395.80555331],
                    [0., 0., 1.]])  # 右相机内参

dist_left = np.array([1.37253265e-01, -4.59574066e-01, -1.81794070e-04, 1.86765804e-03, 4.20309007e-01])  # 左相机畸变系数
dist_right = np.array([1.37253265e-01, -4.59574066e-01, -1.81794070e-04, 1.86765804e-03, 4.20309007e-01])  # 右相机畸变系数

# 设置棋盘格的尺寸，单位：米
inter_corner_shape = (9, 6)  # 11x8的棋盘格
size_per_grid = 0.0914  # 每个方格的尺寸

# 世界坐标系下的角点坐标
world_points = np.array([[x * size_per_grid, y * size_per_grid, 0] for y in range(inter_corner_shape[1]) for x in
                         range(inter_corner_shape[0])], dtype=np.float32)

# 图像路径
img_dir_left = r"/calibration/data/left"
img_dir_right = r"/calibration/data/right"
img_type = "png"  # 图片格式

# 加载图像文件
images_left = sorted(glob.glob(img_dir_left + os.sep + '*.' + img_type))
images_right = sorted(glob.glob(img_dir_right + os.sep + '*.' + img_type))

# 用于存储图像点和世界坐标点
obj_points = []  # 世界坐标系下的角点
img_points_left = []  # 左相机的图像坐标点
img_points_right = []  # 右相机的图像坐标点

# 提取角点
for left_img, right_img in zip(images_left, images_right):
    # 读取图像
    img_left = cv2.imread(left_img)
    img_right = cv2.imread(right_img)

    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # 查找左图和右图中的棋盘格角点
    # ret_left, corners_left = cv2.findChessboardCorners(gray_left, inter_corner_shape, None)
    # ret_right, corners_right = cv2.findChessboardCorners(gray_right, inter_corner_shape, None)

    ret_left, corners_left = cv2.findCirclesGrid(gray_left, inter_corner_shape, None)
    ret_right, corners_right = cv2.findCirclesGrid(gray_right, inter_corner_shape, None)

    # 如果两个图像都成功找到角点
    if ret_left and ret_right:
        obj_points.append(world_points)
        img_points_left.append(corners_left)
        img_points_right.append(corners_right)

        # 可视化角点
        cv2.drawChessboardCorners(img_left, inter_corner_shape, corners_left, ret_left)
        cv2.drawChessboardCorners(img_right, inter_corner_shape, corners_right, ret_right)
        combined_img = np.hstack((img_left, img_right))
        # 显示角点
        cv2.imshow('Left Image', combined_img)
        cv2.waitKey(1)

cv2.destroyAllWindows()

# 执行双目标定
ret, K_left_est, dist_left_est, K_right_est, dist_right_est, R, T, E, F = cv2.stereoCalibrate(
    obj_points,  # 世界坐标点
    img_points_left,  # 左图像的角点
    img_points_right,  # 右图像的角点
    K_left, dist_left,  # 左相机内参和畸变系数
    K_right, dist_right,  # 右相机内参和畸变系数
    gray_left.shape[::-1],  # 图像尺寸
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # 精度和最大迭代次数
)

# 打印计算结果
print("Left Camera Intrinsics: \n", K_left_est)
print("Left Camera Distortion Coefficients: \n", dist_left_est)
print("Right Camera Intrinsics: \n", K_right_est)
print("Right Camera Distortion Coefficients: \n", dist_right_est)
print("Rotation Matrix: \n", R)
print("Translation Vector: \n", T)
print("Essential Matrix: \n", E)
print("Fundamental Matrix: \n", F)
