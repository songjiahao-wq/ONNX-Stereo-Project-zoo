import time
from dataclasses import dataclass
from config import Stereo
import cv2
import numpy as np
import onnxruntime
from stereomodel import BaseONNXInference, BaseTRTInference


class CREStereo_ONNX(BaseONNXInference):

    def __init__(self, model_path):
        super().__init__(model_path)

    def __call__(self, left_img, right_img):
        return self.update(left_img, right_img)


class CREStereo_TRT(BaseTRTInference):

    def __init__(self, model_path):
        super().__init__(model_path)

    def __call__(self, left_img, right_img):
        return self.update(left_img, right_img)

    def process_output(self, outputs):
        disp_pred = outputs.reshape(1, 2, self.height, self.width)  # 确保 shape 正确
        disp_pred = np.squeeze(disp_pred[:, 0, :, :])  # 移除 batch 维度
        return disp_pred
if __name__ == '__main__':
    Stereo = Stereo(res_height=480, res_width=640)
    use_onnx = True
    if use_onnx:
        # Initialize model
        model_path = './weights/crestereo_init_iter2_480x640.onnx'
        depth_estimator = CREStereo_ONNX(model_path)
    else:
        # Initialize model
        model_path = './weights/crestereo_init_iter2_480x640fp32.engine'
        depth_estimator = CREStereo_TRT(model_path)
    # Load images
    left_img = cv2.imread('../../data/640x352/im0.png')
    right_img = cv2.imread('../../data/640x352/im1.png')

    # Estimate depth and colorize it
    for i in range(1):
        disparity_map = depth_estimator(left_img, right_img)
    color_disparity = depth_estimator.draw_disparity()
    Stereo.show_depth_point(disparity_map, left_img)
    combined_img = np.hstack((left_img, color_disparity))

    cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)
    cv2.imshow("Estimated disparity", combined_img)
    cv2.waitKey(0)
