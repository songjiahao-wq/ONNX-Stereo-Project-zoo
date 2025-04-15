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



class IGEVStereo_ONNX(BaseONNXInference):

    def __init__(self, model_path):
        super().__init__(model_path)

    def __call__(self, left_img, right_img):
        return self.update(left_img, right_img)


    def prepare_input(self, img, half=False):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(img, (self.input_width, self.input_height), cv2.INTER_AREA)

        img_input = img_input.transpose(2, 0, 1)
        img_input = img_input[np.newaxis, :, :, :]

        return img_input.astype(np.float32)


class IGEVStereo_TRT(BaseTRTInference):

    def __init__(self, model_path):
        super().__init__(model_path)


    def process_output(self, outputs):
        disp_pred = outputs.reshape(1, 1, self.height, self.width)  # 确保 shape 正确
        disp_pred = np.squeeze(disp_pred[:, 0, :, :])  # 移除 batch 维度
        return disp_pred



if __name__ == '__main__':
    use_onnx = False
    if use_onnx:
        # Initialize model
        model_path = './weights/rt_sceneflow_640480.onnx'
        depth_estimator = IGEVStereo_ONNX(model_path)
    else:
        # Initialize model
        model_path = './weights/rt_sceneflow_640480fp16 copy.engine'
        depth_estimator = IGEVStereo_TRT(model_path)
    # Load images
    left_img = cv2.imread('../../data/mid/im0.png')
    right_img = cv2.imread('../../data/mid/im1.png')

    # Estimate depth and colorize it
    start_time = time.time()
    for i in range(1):
        disparity_map = depth_estimator(left_img, right_img)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    color_disparity = depth_estimator.draw_disparity()

    combined_img = np.hstack((left_img, color_disparity))

    cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)
    cv2.imshow("Estimated disparity", combined_img)
    cv2.waitKey(0)
