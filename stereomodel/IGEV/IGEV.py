# -*- coding: utf-8 -*-
# @Time    : 2025/3/30 18:01
# @Author  : sjh
# @Site    : 
# @File    : hitstereo.py
# @Comment :
import time
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime
from enum import Enum
from stereomodel import BaseONNXInference, BaseTRTInference
from utils.util import get_points_3d, draw_points3d


class IGEVStereo_ONNX(BaseONNXInference):

    def __init__(self, model_path):
        super().__init__(model_path)

    def __call__(self, left_img, right_img):
        return self.update(left_img, right_img)


    def prepare_input(self, img, half=False):
        img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img_input = cv2.resize(img, (self.input_width, self.input_height), cv2.INTER_AREA)

        img_input = img_input.transpose(2, 0, 1)
        img_input = img_input[np.newaxis, :, :, :]

        return img_input.astype(np.float32)


class IGEVStereo_TRT(BaseTRTInference):

    def __init__(self, model_path):
        super().__init__(model_path)


    def process_output(self, outputs):
        disp_pred = outputs.reshape(1, -1, self.height, self.width)  # 确保 shape 正确
        print(disp_pred.shape)
        disp_pred = np.squeeze(disp_pred[:, 0, :, :])  # 移除 batch 维度
        self.disparity_map = disp_pred
        return disp_pred

    def draw_disparity2(self, max_disparity=78, min_disparity=7.8):
        
        # 将负值设置为0（黑色）
        disparity_map = np.where(self.disparity_map < 0, 0, self.disparity_map)
        disparity_map = disparity_map.astype(np.float32)
        
        disparity_map = np.clip(disparity_map, min_disparity, max_disparity)
        # disparity_map = disparity_map[disparity_map < max_disparity]
        # disparity_map = disparity_map[disparity_map > min_disparity]
        # norm_disparity_map = 255 * (disparity_map / (np.max(disparity_map) - np.min(disparity_map)))  # 归一化到[0, 255]
        norm_disparity_map = 255 * (disparity_map / (max_disparity - min_disparity))  # 归一化到[0, 255]

        color_disparity_map = cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map, 1), cv2.COLORMAP_JET)
        
        return color_disparity_map
    
if __name__ == '__main__':
    from config import Stereo

    Stereo = Stereo(res_height=480, res_width=640)
    use_onnx = False
    if use_onnx:
        # Initialize model
        model_path = './weights/igev_sceneflow_HxW.onnx'
        depth_estimator = IGEVStereo_ONNX(model_path)
    else:
        # Initialize model
        model_path = './weights/igev_sceneflow_480x640fp16.engine'
        depth_estimator = IGEVStereo_TRT(model_path)
    # # Load images
    # left_img = cv2.imread('../../data/528x392/im0.png')
    # right_img = cv2.imread('../../data/528x392/im1.png')
    #
    # # Estimate depth and colorize it
    # start_time = time.time()
    # for i in range(1):
    #     disparity_map = depth_estimator(left_img, right_img)
    # end_time = time.time()
    # print(f"Time taken: {end_time - start_time} seconds")
    # # depth_estimator.measure_speed()
    # color_disparity = depth_estimator.draw_disparity()
    # print(color_disparity.shape, left_img.shape)
    # combined_img = np.hstack((left_img, color_disparity))
    #
    # Stereo.show_depth_point(disparity_map, left_img)

    video_path = r"D:\BaiduSyncdisk\work\Stereo\ONNX-Stereo-Project-zoo\calibration\rectified_video.avi"
    cap_video = cv2.VideoCapture(video_path)
    while cap_video.isOpened():
        start_time = time.perf_counter()
        try:
            ret_l, cobined_image = cap_video.read()
        except:
            continue
        if cobined_image is None:
            break
        rectifyed_left = cobined_image[:cobined_image.shape[0]//2, :]
        rectifyed_right = cobined_image[cobined_image.shape[0]//2:, :]
        disparity_map = depth_estimator(rectifyed_left, rectifyed_right)
        color_disparity = depth_estimator.draw_disparity()
        combined_img = np.hstack((rectifyed_left, color_disparity))
        cv2.imshow("Estimated disparity", combined_img)
        cv2.waitKey(0)