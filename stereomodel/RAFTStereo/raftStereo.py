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



class RAFTStereo_ONNX(BaseONNXInference):

    def __init__(self, model_path):
        super().__init__(model_path)
        self.input_width = 640
        self.input_height = 480

    def __call__(self, left_img, right_img):
        return self.update(left_img, right_img)
    def process_output(self, output):
        print(output.shape)
        return np.squeeze(output[:, 0, :, :])

    def prepare_input(self, img, half=False):
        img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(img, (self.input_width, self.input_height), cv2.INTER_AREA)

        img_input = img_input.transpose(2, 0, 1) 
        img_input = img_input[np.newaxis, :, :, :] 

        return img_input.astype(np.float32)


class RAFTStereo_TRT(BaseTRTInference):

    def __init__(self, model_path):
        super().__init__(model_path)
        self.input_width = 640
        self.input_height = 480
    def preprocess_image(self, img_input):
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(img_input, (self.input_width, self.input_height), cv2.INTER_AREA)

        img_input = img_input.transpose(2, 0, 1) 
        img_input = img_input[np.newaxis, :, :, :] 

        return np.ascontiguousarray(img_input.astype(np.float32))
    
    def process_output(self, outputs):
        disp_pred = np.squeeze(outputs[:, 0, :, :])  # 移除 batch 维度
        return disp_pred


if __name__ == '__main__':
    use_onnx = True
    if use_onnx:
        # Initialize model
        model_path = 'weights/raft_sceneflow_iter10_480x640.onnx'
        depth_estimator = RAFTStereo_ONNX(model_path)
    else:
        # Initialize model
        model_path = 'D:/project2024/stereo matching/RAFT-Stereo/raft_sceneflow_iter10_dyfp16.engine'
        depth_estimator = RAFTStereo_TRT(model_path)
    # Load images
    left_img = cv2.imread('../../data/640x352/im0.png').astype(np.uint8)
    right_img = cv2.imread('../../data/640x352/im1.png').astype(np.uint8)

    # Estimate depth and colorize it
    start_time = time.time()
    for i in range(1):
        disparity_map = depth_estimator(left_img, right_img)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    import matplotlib.pyplot as plt
    plt.imsave('disparity.png', -disparity_map, cmap='jet')

    color_disparity = depth_estimator.draw_disparity()

    combined_img = np.hstack((cv2.resize(left_img, (color_disparity.shape[1], color_disparity.shape[0])), color_disparity))

    cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)
    cv2.imshow("Estimated disparity", combined_img)
    cv2.waitKey(0)
