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


class HITStereo_ONNX(BaseONNXInference):

    def __init__(self, model_path):
        super().__init__(model_path)
        self.model_type = ModelType.middlebury
        self.input_width = 640
        self.input_height = 480
    def __call__(self, left_img, right_img):
        return self.update(left_img, right_img)
    def inference_without_flow(self, left_tensor, right_tensor):
        print(left_tensor.shape, right_tensor.shape)
        return self.session.run(self.output_names, {self.input_names[0]: left_tensor,
                                                    self.input_names[1]: right_tensor})[0]
    def np2numpy(self,x, t=True, bgr=False):
        x = cv2.resize(x, (self.input_width, self.input_height))
        
        if len(x.shape) == 2:
            x = x[..., None]  # 将灰度图像扩展为 3 通道（最后一个维度为 1）
        
        if bgr:
            x = x[..., [2, 1, 0]]  # 将 BGR 转换为 RGB
            print(x.shape,'aaaaaaaaaaaaa')
        if t:
            x = np.transpose(x, (2, 0, 1))  # 将 HWC 格式转为 CHW 格式
            print(x.shape,'aaaaaaaaaaaaa')
        print(x.dtype,'aaaaaaaaaaaaa')
        if x.dtype == np.uint8:
            x = x.astype(np.float32)  # 将 uint8 转换为 [0, 1] 范围的 float32
        
        return x
    def prepare_input(self, img):
        img = self.np2numpy(img, t=True, bgr=True) / 255.0
        img = img[np.newaxis, :, :, :]
        print(img.shape,'aaaaaaaaaaaaa')
        return img.astype(np.float32)


    def process_output(self, output):
        return np.squeeze(output)


class HITStereo_TRT(BaseTRTInference):

    def __init__(self, model_path):
        super().__init__(model_path)

    # def update(self, left_image, right_image):
    #     """
    #     进行深度估计推理，返回视差图（深度图）。
    #     """
    #     self.cuda_ctx.push()

    #     self.img_height, self.img_width = left_image.shape[:2]
    #     # **1️⃣ 预处理**
    #     combined_img = self.preprocess_image(left_image, right_image)
    #     outputs = self.inference(combined_img)
    #     # **6️⃣ 解析输出**
    #     self.disparity_map = self.process_output(outputs)
    #     self.cuda_ctx.pop()

    #     return self.disparity_map

    # def preprocess_image(self, left_image, right_image):
    #     left_image = cv2.resize(left_image, (self.width, self.height))
    #     right_image = cv2.resize(right_image, (self.width, self.height))
    #     left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
    #     right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
    #     combined_img = np.concatenate((left_image, right_image), axis=-1) / 255.0
    #     combined_img = np.transpose(combined_img, (2, 0, 1))[None, :, :, :].astype(np.float32)
    #     return np.ascontiguousarray(combined_img)

    # def process_output(self, outputs):
    #     disp_pred = outputs.reshape(1, 1, self.height, self.width)  # 确保 shape 正确
    #     disp_pred = np.squeeze(disp_pred[:, 0, :, :])  # 移除 batch 维度
    #     return disp_pred

    # def inference(self, combined_img):
    #     """
    #     进行深度估计推理，返回视差图（深度图）。
    #     """
    #     # **2️⃣ 传输数据到 GPU**
    #     self.cuda.memcpy_htod(self.inputs[0]["device"], combined_img)
    #     # **3️⃣ 执行 TensorRT 推理**
    #     success = self.context.execute_v2(bindings=self.bindings)
    #     # **4️⃣ 从 GPU 拷贝输出**
    #     self.cuda.memcpy_dtoh(self.outputs[0]["host"], self.outputs[0]["device"])

    #     # **5️⃣ 确保输出有效**
    #     if np.isnan(self.outputs[0]["host"]).sum() > 0:
    #         raise RuntimeError("[ERROR] 推理输出包含 NaN！")
    #     # **6️⃣ 解析输出**
    #     outputs = self.outputs[0]["host"]

    #     return outputs
    def measure_speed(self):
        self.cuda.Context.synchronize()
        start_time = time.time()
        for _ in range(1000):
            self.inference(np.ascontiguousarray(np.zeros((1, 6, self.height, self.width), dtype=np.float32)))
        self.cuda.Context.synchronize()
        end_time = time.time()
        print(f"🔹 推理耗时: {(end_time - start_time)} ms")

if __name__ == '__main__':
    use_onnx = True
    if use_onnx:
        # Initialize model
        model_path = 'weights/stereo_net_480x640_nonopt.onnx'
        depth_estimator = HITStereo_ONNX(model_path)
    else:
        # Initialize model
        # model_path = './weights/model_float32_optfp16.engine'
        # model_path = './weights/model_float32_optfp32.engine'
        model_path = 'weights/hitnet_xl_sf_finalpass_from_tf_480x640fp16.engine'
        # model_path = './weights/model_float32fp32.engine'
        depth_estimator = HITStereo_TRT(model_path)
    # Load images
    left_img = cv2.imread(r'D:\BaiduSyncdisk\work\Stereo\data/111/im0.png', cv2.IMREAD_COLOR)
    right_img = cv2.imread(r'D:\BaiduSyncdisk\work\Stereo\data/111/im1.png', cv2.IMREAD_COLOR)
    disparity_map = depth_estimator(left_img, right_img)
    disparity_map = depth_estimator.draw_disparity()
    cv2.imshow('disparity_map', disparity_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

