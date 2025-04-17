# -*- coding: utf-8 -*-
# @Time    : 2025/4/6 21:45
# @Author  : sjh
# @Site    : 
# @File    : Stereo.py
# @Comment :
import cv2
import numpy as np
import json
import glob
import os
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

class stereoRectify:
    def __init__(self, findChessboardCorners=False):
        
        self.findChessboardCorners = findChessboardCorners
        
        self.image_size = (1056, 784)   # 图像尺寸
        # 设置棋盘格的尺寸，单位：米
        self.board_size = (9, 6)  # 11x8的棋盘格
        self.square_size = 0.0914  # 每个方格的尺寸

        # 世界坐标系下的角点坐标
        # self.world_points = np.array([[x * self.square_size, y * self.square_size, 0] for y in range(self.board_size[1]) for x in range(self.board_size[0])], dtype=np.float32)
        self.world_points = np.zeros((self.board_size[0] * self.board_size[1], 3), dtype=np.float32)
        self.world_points[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2) * self.square_size
        # print("World points (前5个点):\n", self.world_points[:5])
    # 输入: 左右相机拍摄的棋盘格图像（或圆点标定板）
    # 输出: 相机内参（K_left, K_right）、畸变系数（dist_left, dist_right）、外参（R, T）

    def stereo_calibrate_image(self, left_images, right_images):
        # 1. 初始化标定板角点坐标（世界坐标系）
        obj_points = []  # 3D点（棋盘格角点的物理坐标）
        left_img_points = []  # 左图像2D点
        right_img_points = []  # 右图像2D点
        detect_count = 0
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
                flags = (cv2.CALIB_CB_SYMMETRIC_GRID) 
                    #  cv2.CALIB_CB_CLUSTERING )  # 启用聚类优化
                    #  cv2.CALIB_CB_ASYMMETRIC_GRID)  # 适应非对称排列
                ret_left, corners_left = cv2.findCirclesGrid(gray_left, self.board_size, None, flags) 
                ret_right, corners_right = cv2.findCirclesGrid(gray_right, self.board_size, None, flags) 
                detect_count += 1
            if ret_left and ret_right:
                print("detect_count: ", detect_count)
                
                obj_points.append(self.world_points)  # 3D点
                left_img_points.append(corners_left)  # 左图像2D点
                right_img_points.append(corners_right)  # 右图像2D点
                
                # 可视化角点
                cv2.drawChessboardCorners(img_left, self.board_size, corners_left, ret_left)
                cv2.drawChessboardCorners(img_right, self.board_size, corners_right, ret_right)
                combined_img = cv2.resize(np.hstack((img_left, img_right)), (1280, 720))
                cv2.imshow('Left Image', combined_img)
                cv2.waitKey(1)
            else:
                print('false')
        # 3. 单目标定（计算内参和畸变系数）
        ret_left, K_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            obj_points, left_img_points, gray_left.shape[::-1], None, None
        )
        ret_right, K_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            obj_points, right_img_points, gray_right.shape[::-1], None, None
        )
        # OmtxL, roiL = cv2.getOptimalNewCameraMatrix(K_left, dist_left, (gray_left.shape[1], gray_left.shape[0]), 1, (gray_left.shape[1], gray_left.shape[0]))
        # OmtxR, roiR = cv2.getOptimalNewCameraMatrix(K_right, dist_right, (gray_right.shape[1], gray_right.shape[0]), 1, (gray_right.shape[1], gray_right.shape[0]))
        # 4. 双目标定（计算旋转矩阵 R 和平移向量 T）
        ret, K_left_est, dist_left_est, K_right_est, dist_right_est, R, T, E, F = cv2.stereoCalibrate(
            obj_points,  # 世界坐标点
            left_img_points,  # 左图像的角点
            right_img_points,  # 右图像的角点
            K_left, dist_left,  # 左相机内参和畸变系数
            K_right, dist_right,  # 右相机内参和畸变系数
            self.image_size,  # 图像尺寸
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)  # 精度和最大迭代次数
        )
        print(K_left, K_left_est)
        print(K_right, K_right_est)
        print(dist_left, dist_left_est)
        print(dist_right, dist_right_est)
        left_reprojection_error, left_error_list = self.compute_reprojection_error(obj_points, left_img_points, rvecs_left, tvecs_left, K_left_est, dist_left_est)
        right_reprojection_error, right_error_list = self.compute_reprojection_error(obj_points, right_img_points, rvecs_right, tvecs_right, K_right_est, dist_right_est)
        print("Left Reprojection Error: \n", left_reprojection_error)
        print("Right Reprojection Error: \n", right_reprojection_error)
        # # 可视化左右误差列表在一张图里，横坐标为图像索引，纵坐标为误差，每个点用一个圆圈表示
        # # 输出最大误差的图像名称
        # max_error_index = np.argmax(left_error_list)
        # print("Max Error Image: \n", left_images[max_error_index])
        # max_error_index = np.argmax(right_error_list)
        # print("Max Error Image: \n", right_images[max_error_index])
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.plot(left_error_list, 'o')
        # plt.xlabel('Image Index')
        # plt.ylabel('Reprojection Error')
        # plt.title('Left Reprojection Error')
        # plt.subplot(1, 2, 2)
        # plt.plot(right_error_list, 'o')
        # plt.xlabel('Image Index')
        # plt.ylabel('Reprojection Error')
        # plt.title('Right Reprojection Error')
        # plt.show()


        # 5. 计算校正变换矩阵
        _, _, _, _, Q = self.stereo_rectify_without_distortion(K_left_est, K_right_est, R, T, gray_left.shape[::-1])
        # _, _, _, _, Q = self.stereo_rectify_with_distortion(K_left_est, dist_left_est, K_right_est, dist_right_est, R, T,self.image_size)
        
        # 打印计算结果
        print("Left Camera Intrinsics: \n", K_left_est)
        print("Left Camera Distortion Coefficients: \n", dist_left_est)
        print("Right Camera Intrinsics: \n", K_right_est)
        print("Right Camera Distortion Coefficients: \n", dist_right_est)
        print("Rotation Matrix: \n", R)
        print("Translation Vector: \n", T)
        print("Essential Matrix: \n", E)
        print("Fundamental Matrix: \n", F)
        print("Q Matrix: \n", Q)
        self.calculate_fx_fy_cx_cy_baseline(np.array(Q))
        print("fx: \n", self.fx)
        print("fy: \n", self.fy)
        print("cx: \n", self.cx)
        print("cy: \n", self.cy)
        print("baseline: \n", self.baseline)
        return K_left, dist_left, K_right, dist_right, R, T, Q


    def stereo_calibrate_video(self, left_video_path, right_video_path, error_threshold=2.0):
        # 1. 初始化标定板角点坐标（世界坐标系）
        obj_points = []  # 3D点（棋盘格角点的物理坐标）
        left_img_points = []  # 左图像2D点
        right_img_points = []  # 右图像2D点
        detect_count = 0
        cap_left = cv2.VideoCapture(left_video_path)
        cap_right = cv2.VideoCapture(right_video_path)
        # 2. 检测棋盘格角点
        frame_count = 0
        while True:
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()
            if not ret_left or not ret_right:
                break
            frame_count += 1
            if frame_count % 20== 0 and detect_count < 100:
                gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
                gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
                flags = (cv2.CALIB_CB_SYMMETRIC_GRID) 
                #  cv2.CALIB_CB_CLUSTERING )  # 启用聚类优化
                #  cv2.CALIB_CB_ASYMMETRIC_GRID)  # 适应非对称排列
                ret_left, corners_left = cv2.findCirclesGrid(gray_left, self.board_size, None) 
                ret_right, corners_right = cv2.findCirclesGrid(gray_right, self.board_size, None)
                remove_index_left =   [7, 11, 14, 32, 38]
                remove_index_right =  [7, 11, 14, 26, 32, 38]
                if ret_left and ret_right:
                    # if detect_count in remove_index_left or detect_count in remove_index_right:
                    #     detect_count += 1
                    #     continue
                    
                    detect_count += 1
                    print("detect_count: ", detect_count)

                    obj_points.append(self.world_points)  # 3D点
                    left_img_points.append(corners_left)  # 左图像2D点
                    right_img_points.append(corners_right)  # 右图像2D点
                    
                    # 可视化角点
                    cv2.drawChessboardCorners(frame_left, self.board_size, corners_left, ret_left)
                    cv2.drawChessboardCorners(frame_right, self.board_size, corners_right, ret_right)
                    combined_img = cv2.resize(np.hstack((frame_left, frame_right)), (1280, 720))
                    cv2.imshow('Left Image', combined_img)
                    cv2.waitKey(1)
                else:
                    print('false')
        cv2.destroyAllWindows()
        # 3. 单目标定（计算内参和畸变系数）
        ret_left, K_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            obj_points, left_img_points, gray_left.shape[::-1], None, None#, flags=cv2.CALIB_USE_INTRINSIC_GUESS
        )
        ret_right, K_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            obj_points, right_img_points, gray_right.shape[::-1], None, None
        )
        # OmtxL, roiL = cv2.getOptimalNewCameraMatrix(K_left, dist_left, self.image_size, alpha=1, newImgSize=self.image_size)
        # OmtxR, roiR = cv2.getOptimalNewCameraMatrix(K_right, dist_right,  self.image_size, alpha=0, newImgSize=self.image_size)
        # 4. 双目标定（计算旋转矩阵 R 和平移向量 T）
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            obj_points,  # 世界坐标点
            left_img_points,  # 左图像的角点
            right_img_points,  # 右图像的角点
            K_left, dist_left,  # 左相机内参和畸变系数
            K_right, dist_right,  # 右相机内参和畸变系数
            self.image_size,  # 图像尺寸
            # criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)  # 精度和最大迭代次数
        )
        left_reprojection_error, left_error_list = self.compute_reprojection_error(obj_points, left_img_points, rvecs_left, tvecs_left, K_left, dist_left)
        right_reprojection_error, right_error_list = self.compute_reprojection_error(obj_points, right_img_points, rvecs_right, tvecs_right, K_right, dist_right)
        print("Left Reprojection Error: \n", left_reprojection_error)
        print("Right Reprojection Error: \n", right_reprojection_error)
        # 可视化左右误差列表在一张图里，横坐标为图像索引，纵坐标为误差，每个点用一个圆圈表示
        # 输出最大误差的图像名称
        max_error_index = np.argmax(left_error_list)
        print("Max Error Image left: \n", max_error_index)
        max_error_index = np.argmax(right_error_list)
        print("Max Error Image right: \n", max_error_index)
        # 输出误差大于2的index
        left_error_index = []
        right_error_index = []
        for i in range(len(left_error_list)):
            if left_error_list[i] > 2:
                left_error_index.append(i)
            if right_error_list[i] > 2:
                right_error_index.append(i)
        print("Left Error Index: \n", left_error_index)
        print("Right Error Index: \n", right_error_index)
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.plot(left_error_list, 'o')
        # plt.xlabel('Image Index')
        # plt.ylabel('Reprojection Error')
        # plt.title('Left Reprojection Error')
        # plt.subplot(1, 2, 2)
        # plt.plot(right_error_list, 'o')
        # plt.xlabel('Image Index')
        # plt.ylabel('Reprojection Error')
        # plt.title('Right Reprojection Error')
        # plt.show()


        # 5. 计算校正变换矩阵
        # _, _, _, _, Q = self.stereo_rectify_without_distortion(K_left, K_right, R, T, self.image_size)
        _, _, _, _, Q = self.stereo_rectify_with_distortion(K_left, dist_left, K_right, dist_right, R, T,self.image_size)
        
        # 打印计算结果
        print("Left Camera Intrinsics: \n", K_left)
        print("Left Camera Distortion Coefficients: \n", dist_left)
        print("Right Camera Intrinsics: \n", K_right)
        print("Right Camera Distortion Coefficients: \n", dist_right)
        print("Rotation Matrix: \n", R)
        print("Translation Vector: \n", T)
        print("Essential Matrix: \n", E)
        print("Fundamental Matrix: \n", F)
        print("Q Matrix: \n", Q)
        self.calculate_fx_fy_cx_cy_baseline(np.array(Q))
        print("fx=", self.fx)
        print("fy=", self.fy)
        print("cx=", self.cx)
        print("cy=", self.cy)
        print("baseline=", self.baseline)
        return K_left, dist_left, K_right, dist_right, R, T, Q
    # 计算重投影误差
    def compute_reprojection_error(self, object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs):
        """
        对所有图像计算重投影均方根误差(RMSE)
        """
        total_error = 0
        total_points = 0
        error_list = []
        for i in range(len(object_points)):
            # 将3D点投影到图像平面
            imgpoints2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            # 计算投影点与实际检测到的角点之间的欧氏距离
            error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2)
            total_error += error**2
            total_points += len(object_points[i])
            error_list.append(error)
        mean_error = np.sqrt(total_error / total_points)
        return mean_error, error_list

    def compute_reprojection_error_single(self, object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs):
        """
        计算所有图像的重投影均方根误差(RMSE)
        """
        total_error = 0
        total_points = 0
        for i in range(len(object_points)):
            # 将3D点投影到图像平面
            imgpoints2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)

            # 计算投影点与实际检测到的角点之间的欧氏距离
            error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2)

            total_error += error ** 2
            total_points += len(object_points[i])

        mean_error = np.sqrt(total_error / total_points)
        return mean_error
    # 计算有畸变立体校正变换
    def stereo_rectify_with_distortion(self, K_left, dist_left, K_right, dist_right, R, T, image_size):

        # 图像尺寸
        gdc_width, gdc_height = self.image_size    # 原始图像尺寸
        out_width, out_height = self.image_size     # 校正后图像尺寸

        # 调用stereoRectify
        flags = cv2.CALIB_ZERO_DISPARITY    # 强制校正后主点垂直对齐
        alpha = 0                           # 自动裁剪黑边
        
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                cameraMatrix1=K_left, # 左相机内参
                distCoeffs1=dist_left, # 左相机畸变系数
                cameraMatrix2=K_right, # 右相机内参
                distCoeffs2=dist_right, # 右相机畸变系数
                imageSize=(gdc_width, gdc_height), # 原始图像尺寸
                R=R, # 旋转矩阵 右到左的旋转矩阵
                T=T, # 平移矩阵 右到左的平移向量
                flags=flags, # 标志
                alpha=alpha, # 自动裁剪黑边
                newImageSize=(out_width, out_height) # 校正后图像尺寸
            )
        # 2. 计算映射表（考虑畸变）
        map1x, map1y = cv2.initUndistortRectifyMap(K_left, dist_left, R1, P1, self.image_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K_right, dist_right, R2, P2, self.image_size, cv2.CV_32FC1)
        
        return map1x, map1y, map2x, map2y, Q
    
    # 计算无畸变立体校正变换
    def stereo_rectify_without_distortion(self, K_left, K_right, R, T, image_size):
        # 1. 计算校正变换矩阵
        print(K_left, K_right, R, T, image_size)
        flags = cv2.CALIB_ZERO_DISPARITY
        alpha = 0
        newImageSize = image_size
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K_left, None, K_right, None, image_size, R, T, flags=flags, alpha=alpha, newImageSize=newImageSize
        )
        # 2. 计算映射表（无畸变，仅校正）
        map1x, map1y = cv2.initUndistortRectifyMap(K_left, None, R1, P1, image_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K_right, None, R2, P2, image_size, cv2.CV_32FC1)
        
        return map1x, map1y, map2x, map2y, Q

    # 计算fx,fy,cx,cy,baseline
    def calculate_fx_fy_cx_cy_baseline(self, Q):
        Q = np.array(Q)
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
    def save_calibration_to_json(self, k_left, dist_left, k_right, dist_right, R, T, Q, E=None, F=None):
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
        k_left, dist_left, k_right, dist_right, R, T, Q = calibration_data['k_left'], calibration_data['dist_left'], calibration_data['k_right'], calibration_data['dist_right'], calibration_data['R'], calibration_data['T'], calibration_data['Q']
        self.calculate_fx_fy_cx_cy_baseline(Q)
        return k_left, dist_left, k_right, dist_right, R, T, Q
    
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

if __name__ == "__main__":
    stereo_rectify = stereoRectify(findChessboardCorners=False)
    method = 3
    if method ==1:
        # 图像路径
        img_dir_left = r"left"
        img_dir_right = r"right"
        img_type = "png"  # 图片格式

        # 加载图像文件
        images_left = sorted(glob.glob(img_dir_left + os.sep + '*.' + img_type))
        images_right = sorted(glob.glob(img_dir_right + os.sep + '*.' + img_type))
        k_left, dist_left, k_right, dist_right, R, T, Q = stereo_rectify.stereo_calibrate(images_left, images_right)
        stereo_rectify.save_calibration_to_json(k_left, dist_left, k_right, dist_right, R, T, Q)
        
        # 矫正一张图
        img_left = cv2.imread(images_left[0])
        img_right = cv2.imread(images_right[0])
        map1x, map1y, map2x, map2y, Q = stereo_rectify.stereo_rectify_without_distortion(k_left, k_right, R, T, stereo_rectify.image_size)
        rectified_left, rectified_right = stereo_rectify.apply_rectification(img_left, img_right, map1x, map1y, map2x, map2y)
        combined_img = cv2.resize(np.hstack((rectified_left, rectified_right)), (1280, 720))
        
        cv2.imshow('Rectified Left', combined_img)
        cv2.waitKey(0)
        # 读取参数
        # k_left, dist_left, k_right, dist_right, R, T, Q = stereo_rectify.load_calibration_from_json('cali_circle.json')
        # print("calibration_data: \n", k_left, dist_left, k_right, dist_right, R, T, Q)   

    elif method == 2:
        left_video_path = r"../calibration/data/20250306134745\step1-3/a0.avi"
        right_video_path = r"../calibration/data/20250306134745\step1-3/a1.avi"
        
        # left_video_path = r"../calibration/data/20250416155006\step1/a0.avi"
        # right_video_path = r"../calibration/data/20250416155006\step1/a1.avi"
        k_left, dist_left, k_right, dist_right, R, T, Q = stereo_rectify.stereo_calibrate_video(left_video_path, right_video_path)
        stereo_rectify.save_calibration_to_json(k_left, dist_left, k_right, dist_right, R, T, Q)


    elif method == 3:
        k_left, dist_left, k_right, dist_right, R, T, Q = stereo_rectify.load_calibration_from_json(r'cali_circle.json')
        print(stereo_rectify.fx, stereo_rectify.fy, stereo_rectify.cx, stereo_rectify.cy, stereo_rectify.baseline)
        k_left, k_right, R, T, Q, dist_left, dist_right = np.array(k_left), np.array(k_right), np.array(R), np.array(T), np.array(Q), np.array(dist_left), np.array(dist_right)
        # 矫正视频
        left_video_path = r"../calibration/data/20250306134745\step2-1/a0.avi"
        right_video_path = r"../calibration/data/20250306134745\step2-1/a1.avi"
        
        # left_video_path = r"../calibration\data\20250305144726\step2-5/a0.avi"
        # right_video_path = r"../calibration\data\20250305144726\step2-5/a1.avi"
        
        left_video_path = r"../calibration/data/20250417104342\step2/a0.avi"
        right_video_path = r"../calibration/data/20250417104342\step2/a1.avi"
        
        cap_left = cv2.VideoCapture(left_video_path)
        cap_right = cv2.VideoCapture(right_video_path)
        
        
        # 保存为ffv1视频
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')
        out = cv2.VideoWriter('rectified_video.avi', fourcc, 30.0, (1056, 784 * 2))
        while True:
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()
            # frame_left = cv2.flip(frame_left, 1)
            # frame_right = cv2.flip(frame_right, 1)
            if not ret_left or not ret_right:
                break
            map1x, map1y, map2x, map2y, _ = stereo_rectify.stereo_rectify_with_distortion(k_left, dist_left, k_right, dist_right, R, T, stereo_rectify.image_size)
            # map1x, map1y, map2x, map2y, _ = stereo_rectify.stereo_rectify_without_distortion(k_left, k_right, R, T, stereo_rectify.image_size)
                    # 畸变矫正
            frame_left = cv2.undistort(frame_left, k_left, dist_left)
            frame_right = cv2.undistort(frame_right, k_right, dist_right)
            rectified_left, rectified_right = stereo_rectify.apply_rectification(frame_left, frame_right, map1x, map1y, map2x, map2y)
            combined_img = np.vstack((rectified_left, rectified_right))
            print(frame_left.shape, frame_right.shape)
            print(combined_img.shape)
            out.write(combined_img)
            cv2.imshow('Rectified Left', combined_img)
            # cv2.waitKey(1)
            # 使用sgbm得到视差图
            # print(rectified_left.shape, rectified_right.shape)
            # gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
            # gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)
            # 左右图划多个线线判断是否对齐
            for i in range(10):
                cv2.line(rectified_left, (0, 100 + i * 50), (1056, 100 + i * 50), (0, 0, 255), 1)
                cv2.line(rectified_right, (0, 100 + i * 50), (1056, 100 + i * 50), (0, 0, 255), 1)
            combined_img = np.hstack((rectified_left, rectified_right))
            cv2.imshow('Rectified Right', combined_img)
            cv2.waitKey(1)
        out.release()