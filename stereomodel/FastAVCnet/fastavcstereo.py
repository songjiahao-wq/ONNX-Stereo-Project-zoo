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


class ModelType(Enum):
    eth3d = 0
    middlebury = 1
    flyingthings = 2


class FastAVCnetStereo_ONNX(BaseONNXInference):

    def __init__(self, model_path):
        super().__init__(model_path)
        self.model_type = ModelType.middlebury

    def __call__(self, left_img, right_img):
        return self.update(left_img, right_img)


    def prepare_input(self, img, half=False):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(img, (self.input_width, self.input_height), cv2.INTER_AREA)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img_input = ((img_input / 255.0 - mean) / std)
        img_input = img_input.transpose(2, 0, 1)
        img_input = img_input[np.newaxis, :, :, :]

        return img_input.astype(np.float32)


class FastAVCnetStereo_TRT(BaseTRTInference):

    def __init__(self, model_path):
        super().__init__(model_path)


    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height))
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        image = (image / 255.0 - mean) / std
        image = image.transpose(2, 0, 1)
        image = image[np.newaxis, :, :, :]
        return np.ascontiguousarray(image.astype(np.float32))

    def process_output(self, outputs):
        disp_pred = outputs.reshape(1, 1, self.height, self.width)  # 确保 shape 正确
        disp_pred = np.squeeze(disp_pred[:, 0, :, :])  # 移除 batch 维度
        return disp_pred



if __name__ == '__main__':
    use_onnx = False
    if use_onnx:
        # Initialize model
        model_path = './weights/fast_acvnet_sceneflow_opset16_480x640.onnx'
        depth_estimator = FastAVCnetStereo_ONNX(model_path)
    else:
        # Initialize model
        model_path = './weights/fast_acvnet_sceneflow_opset16_480x640fp16.engine'
        depth_estimator = FastAVCnetStereo_TRT(model_path)
    # Load images
    left_img = cv2.imread('../../data/mid/im0.png')
    right_img = cv2.imread('../../data/mid/im1.png')

    # Estimate depth and colorize it
    for i in range(1):
        disparity_map = depth_estimator(left_img, right_img)
    color_disparity = depth_estimator.draw_disparity()

    combined_img = np.hstack((left_img, color_disparity))

    cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)
    cv2.imshow("Estimated disparity", combined_img)
    cv2.waitKey(0)
