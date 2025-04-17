# -*- coding: utf-8 -*-
"""
Homework: Calibrate the Camera with ZhangZhengyou Method.
Picture File Folder: ".\pic\IR_camera_calib_img", With Distort. 

By YouZhiyuan 2019.11.18
"""

import os
import numpy as np
import cv2
import glob


def calib(inter_corner_shape, size_per_grid, video_path):
    # criteria: only for subpix calibration, which is not used here.
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    w,h = inter_corner_shape
    # cp_int: corner point in int form, save the coordinate of corner points in world sapce in 'int' form
    # like (0,0,0), (1,0,0), (2,0,0) ....,(10,7,0).
    cp_int = np.zeros((w*h,3), np.float32)
    cp_int[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    # cp_world: corner point in world space, save the coordinate of corner points in world space.
    cp_world = cp_int*size_per_grid
    
    obj_points = [] # the points in world space
    img_points = [] # the points in image space (relevant to obj_points)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    detect_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 20 == 0 and detect_count < 40:
            detect_count += 1
            print(f"Detecting... {detect_count}th frame")
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Find corners for chessboard or circle grid (commented out chessboard code)
            ret, cp_img = cv2.findCirclesGrid(gray_img, (w, h), None)

            if ret:
                obj_points.append(cp_world)
                img_points.append(cp_img)

                # Optional: Show the detected corners
                cv2.drawChessboardCorners(frame, (w, h), cp_img, ret)
                cv2.imshow('FoundCorners', frame)
                cv2.waitKey(1)
    cv2.destroyAllWindows()
    # calibrate the camera
    ret, mat_inter, coff_dis, v_rot, v_trans = cv2.calibrateCamera(obj_points, img_points, gray_img.shape[::-1], None, None)
    if not ret:
        print("Error: Calibration failed.")
        return None, None

    print(f"ret: {ret}")
    print(f"Internal Matrix (Camera Matrix):\n{mat_inter}")
    print(f"Distortion Coefficients:\n{coff_dis}")
    # print(f"Rotation Vectors:\n{v_rot}")
    # print(f"Translation Vectors:\n{v_trans}")

    # Calculate the reproject error
    total_error = 0
    for i in range(len(obj_points)):
        img_points_repro, _ = cv2.projectPoints(obj_points[i], v_rot[i], v_trans[i], mat_inter, coff_dis)
        error = cv2.norm(img_points[i], img_points_repro, cv2.NORM_L2) / len(img_points_repro)
        print(f"error: {error}")
        total_error += error
    print(f"Average Reprojection Error: {total_error / len(obj_points)}")

    return mat_inter, coff_dis
    
def dedistortion(inter_corner_shape, video_path, mat_inter, coff_dis):
    w,h = inter_corner_shape
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = frame
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mat_inter,coff_dis,(w,h),0,(w,h)) # 自由比例参数
        dst = cv2.undistort(img, mat_inter, coff_dis, None, newcameramtx)
        # clip the image
        # x,y,w,h = roi
        # dst = dst[y:y+h, x:x+w]
        combined = np.vstack((img, dst))
        cv2.imshow('dst', combined)
        cv2.waitKey(0)
    print('Dedistorted images have been saved to: %s successfully.' %save_dir)
    
if __name__ == '__main__':
    inter_corner_shape = (9,6)
    size_per_grid = 0.0914
    img_dir = "a1"
    video_path = r"../calibration/data/20250416155006/step1/a0.avi"
    # calibrate the camera
    mat_inter, coff_dis = calib(inter_corner_shape, size_per_grid, video_path)
    
    
    
    
    # dedistort and save the dedistortion result. 
    # save_dir = "data/save_dedistortion"
    # if(not os.path.exists(save_dir)):
        # os.makedirs(save_dir)
    # dedistortion(inter_corner_shape, video_path, mat_inter, coff_dis)
    
    
    