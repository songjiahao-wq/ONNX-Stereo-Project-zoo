# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import glob


def calib(inter_corner_shape, size_per_grid, img_dir,img_type):
    # criteria: only for subpix calibration, which is not used here.
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    w, h = inter_corner_shape
    # cp_int: corner point in int form, save the coordinate of corner points in world sapce in 'int' form
    # like (0,0,0), (1,0,0), (2,0,0) ....,(10,7,0).
    cp_int = np.zeros((w*h, 3), np.float32)
    cp_int[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    # cp_world: corner point in world space, save the coordinate of corner points in world space.
    cp_world = cp_int*size_per_grid
    
    obj_points = [] # the points in world space
    img_points = [] # the points in image space (relevant to obj_points)
    images = glob.glob(img_dir + os.sep + '**.' + img_type)
    for fname in images:
        img = cv2.imread(fname)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # find the corners, cp_img: corner points in pixel space.
        # ret, cp_img = cv2.findChessboardCorners(gray_img, (w, h), None)
        ret, cp_img = cv2.findCirclesGrid(gray_img, (w, h), None)
        # if ret is True, save.
        if ret:
            # cv2.cornerSubPix(gray_img,cp_img,(11,11),(-1,-1),criteria)
            obj_points.append(cp_world)
            img_points.append(cp_img)
            # view the corners
            cv2.drawChessboardCorners(img, (w, h), cp_img, ret)
            cv2.imshow('FoundCorners', img)
            cv2.waitKey(1)
    cv2.destroyAllWindows()

    print(obj_points)
    print(img_points)

    # calibrate the camera
    ret, mat_inter, coff_dis, v_rot, v_trans = cv2.calibrateCamera(obj_points, img_points, gray_img.shape[::-1], None, None)
    print("ret:", ret)
    print("internal matrix:\n", mat_inter)
    # in the form of (k_1,k_2,p_1,p_2,k_3)
    print("distortion cofficients:\n", coff_dis)
    print("rotation vectors:\n", v_rot)
    print("translation vectors:\n", v_trans)
    # calculate the error of reproject
    total_error = 0
    for i in range(len(obj_points)):
        img_points_repro, _ = cv2.projectPoints(obj_points[i], v_rot[i], v_trans[i], mat_inter, coff_dis)
        error = cv2.norm(img_points[i], img_points_repro, cv2.NORM_L2)/len(img_points_repro)
        total_error += error
    print("Average Error of Reproject: ", total_error / len(obj_points))
    
    return mat_inter, coff_dis
    
if __name__ == '__main__':
    inter_corner_shape = (9,6)
    size_per_grid = 0.0914
    img_dir = "data/left"
    img_type = "png"
    calib(inter_corner_shape, size_per_grid, img_dir, img_type)
    