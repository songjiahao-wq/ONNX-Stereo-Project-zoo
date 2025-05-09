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


class CGIStereo_ONNX(BaseONNXInference):

    def __init__(self, model_path):
        super().__init__(model_path)
        self.model_type = ModelType.middlebury
        self.input_width, self.input_height = 640, 352
        print(self.input_width, self.input_height)
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
    def process_output(self, output):
        print(output.shape)
        return np.squeeze(output[0, :, :])

class CGIStereo_TRT(BaseTRTInference):

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
    from config import Stereo
    Stereo = Stereo(res_height=480, res_width=640)
    use_onnx = True
    use_video = False
    if use_onnx:
        # Initialize model
        model_path = './weights/cgi_stereo_sceneflow_HxW.onnx'
        depth_estimator = CGIStereo_ONNX(model_path)
    else:
        # Initialize model
        model_path = './weights/fast_acvnet_sceneflow_opset16_480x640fp16.engine'
        depth_estimator = CGIStereo_TRT(model_path)
    if not use_video:
        # Load images
        left_img = cv2.imread('../../data/640x352/im0.png')
        right_img = cv2.imread('../../data/640x352/im1.png')

        # Estimate depth and colorize it
        for i in range(1):
            disparity_map = depth_estimator(left_img, right_img)
        Stereo.show_depth_point(disparity_map, left_img)
        
        color_disparity = depth_estimator.draw_disparity()

        combined_img = np.hstack((left_img, color_disparity))
        cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)
        cv2.imshow("Estimated disparity", combined_img)
        cv2.waitKey(0)
    else:
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
            color_disparity = depth_estimator.draw_disparity()
            combined_img = np.hstack((rectifyed_left, color_disparity))
            cv2.imshow("Estimated disparity", combined_img)
            cv2.waitKey(0)
