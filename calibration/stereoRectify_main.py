# -*- coding: utf-8 -*-
# @Time    : 2025/4/6 21:45
# @Author  : sjh
# @Site    : 
# @File    : Stereo.py
# @Comment :
import cv2
import numpy as np
import json


class stereoRectify:
    def __init__(self, left_img, right_img, findChessboardCorners=True):

        self.findChessboardCorners = findChessboardCorners

        self.image_size = (1920, 1080)  # 图像尺寸
        # 设置棋盘格的尺寸，单位：米
        self.board_size = (9, 6)  # 11x8的棋盘格
        self.square_size = 0.0914  # 每个方格的尺寸

        # 世界坐标系下的角点坐标
        self.world_points = np.array([[x * self.square_size, y * self.square_size, 0] for y in range(self.board_size[1])
                                      for x in range(self.board_size[0])], dtype=np.float32)

    # 输入: 左右相机拍摄的棋盘格图像（或圆点标定板）
    # 输出: 相机内参（K_left, K_right）、畸变系数（dist_left, dist_right）、外参（R, T）

    def stereo_calibrate(self, left_images, right_images):
        # 1. 初始化标定板角点坐标（世界坐标系）
        obj_points = []  # 3D点（棋盘格角点的物理坐标）
        left_img_points = []  # 左图像2D点
        right_img_points = []  # 右图像2D点

        # 2. 检测棋盘格角点
        for left_img, right_img in zip(left_images, right_images):
            img_left = cv2.imread(left_img)
            img_right = cv2.imread(right_img)
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
            if self.findChessboardCorners:
                # 检测左图像角点
                ret_left, corners_left = cv2.findChessboardCorners(gray_left, self.board_size)
                # 检测右图像角点
                ret_right, corners_right = cv2.findChessboardCorners(gray_right, self.board_size)
            else:
                corners_left = cv2.findCirclesGrid(gray_left, self.board_size, None)
                corners_right = cv2.findCirclesGrid(gray_right, self.board_size, None)

            if ret_left and ret_right:
                obj_points.append(self.world_points)  # 3D点
                left_img_points.append(corners_left)  # 左图像2D点
                right_img_points.append(corners_right)  # 右图像2D点

                # 可视化角点
                cv2.drawChessboardCorners(img_left, self.board_size, corners_left, ret_left)
                cv2.drawChessboardCorners(img_right, self.board_size, corners_right, ret_right)
                combined_img = np.hstack((img_left, img_right))
                cv2.imshow('Left Image', combined_img)
                cv2.waitKey(1)

        # 3. 单目标定（计算内参和畸变系数）
        ret_left, K_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            obj_points, left_img_points, left_img.shape[::-1], None, None
        )
        ret_right, K_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            obj_points, right_img_points, right_img.shape[::-1], None, None
        )

        # 4. 双目标定（计算旋转矩阵 R 和平移向量 T）
        ret, K_left_est, dist_left_est, K_right_est, dist_right_est, R, T, E, F = cv2.stereoCalibrate(
            obj_points,  # 世界坐标点
            left_img_points,  # 左图像的角点
            right_img_points,  # 右图像的角点
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
        return K_left, dist_left, K_right, dist_right, R, T

    # 计算有畸变立体校正变换
    def stereo_rectify_with_distortion(self, K_left, dist_left, K_right, dist_right, R, T, image_size):
        # 1. 计算校正变换矩阵
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K_left, dist_left, K_right, dist_right, image_size, R, T
        )
        # 2. 计算映射表（考虑畸变）
        map1x, map1y = cv2.initUndistortRectifyMap(K_left, dist_left, R1, P1, image_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K_right, dist_right, R2, P2, image_size, cv2.CV_32FC1)

        return map1x, map1y, map2x, map2y, Q

    # 计算无畸变立体校正变换
    def stereo_rectify_without_distortion(self, K_left, K_right, R, T, image_size):
        # 1. 计算校正变换矩阵
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K_left, None, K_right, None, image_size, R, T
        )
        # 2. 计算映射表（无畸变，仅校正）
        map1x, map1y = cv2.initUndistortRectifyMap(K_left, None, R1, P1, image_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K_right, None, R2, P2, image_size, cv2.CV_32FC1)

        return map1x, map1y, map2x, map2y, Q

    # 计算fx,fy,cx,cy,baseline
    def calculate_fx_fy_cx_cy_baseline(self, Q):
        self.cx = -Q[0, 3]
        self.cy = -Q[1, 3]
        self.fx = Q[2, 3]
        self.fy = Q[2, 3]
        self.baseline = abs(1.0 / Q[3, 2])
        # return self.fx, self.fy, self.cx, self.cy, self.baseline

    # 应用校正变换
    def apply_rectification(self, left_img, right_img, map1x, map1y, map2x, map2y):
        # 1. 重映射
        rectified_left = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR)

        return rectified_left, rectified_right

    # 保存参数
    def save_calibration_to_json(self, k_left, dist_left, k_right, dist_right, R, T, E, F, Q):
        calibration_data = {
            "k_left": k_left.tolist(),
            "dist_left": dist_left.tolist(),
            "k_right": k_right.tolist(),
            "dist_right": dist_right.tolist(),
            "R": R.tolist(),
            "T": T.tolist(),
            "Q": Q.tolist()
        }

        # 将字典保存为JSON文件
        with open('cali_circle.json', "w") as json_file:
            json.dump(calibration_data, json_file, indent=4)

    # 读取参数
    def load_calibration_from_json(self, file_path):
        with open(file_path, 'r') as json_file:
            calibration_data = json.load(json_file)
        return calibration_data

    # 可视化视差图
    def visualize_disparity_map(self, disparity_map):
        min_depth, max_depth = 1, 10
        min_disparity = self.fx / float(min_depth * self.baseline)
        max_disparity = self.fx / float(max_depth * self.baseline)

        disparity_map = np.where(disparity_map < 0, 0, disparity_map)
        disparity_map = disparity_map.astype(np.float32)

        disparity_map = np.clip(disparity_map, min_disparity, max_disparity)
        norm_disparity_map = 255 * (disparity_map / (max_disparity - min_disparity))  # 归一化到[0, 255]

        color_disparity_map = cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map, 1), cv2.COLORMAP_JET)
        return color_disparity_map
