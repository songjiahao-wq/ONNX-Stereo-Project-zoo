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

    def __call__(self, left_img, right_img):
        return self.update(left_img, right_img)

    def update(self, left_img, right_img):

        input_tensor = self.prepare_input(left_img, right_img)

        # Perform inference on the image
        if self.model_type == ModelType.flyingthings:
            left_disparity, right_disparity = self.inference(input_tensor)
            self.disparity_map = left_disparity
        else:
            self.disparity_map = self.inference(input_tensor)

        # Estimate depth map from the disparity
        self.depth_map = self.get_depth_from_disparity(self.disparity_map, self.camera_config)

        return self.disparity_map

    def inference(self, input_tensor):

        input_name = self.session.get_inputs()[0].name
        left_output_name = self.session.get_outputs()[0].name

        if self.model_type is not ModelType.flyingthings:
            left_disparity = self.session.run([left_output_name], {input_name: input_tensor})
            return np.squeeze(left_disparity)

        right_output_name = self.session.get_outputs()[1].name
        left_disparity, right_disparity = self.session.run([left_output_name, right_output_name], {
            input_name: input_tensor})

        return np.squeeze(left_disparity), np.squeeze(right_disparity)

    def prepare_input(self, left_img, right_img):

        self.img_height, self.img_width = left_img.shape[:2]

        left_img = cv2.resize(left_img, (self.input_width, self.input_height))
        right_img = cv2.resize(right_img, (self.input_width, self.input_height))

        if (self.model_type is ModelType.eth3d):

            # Shape (1, 2, None, None)
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

            left_img = np.expand_dims(left_img, 2)
            right_img = np.expand_dims(right_img, 2)

            combined_img = np.concatenate((left_img, right_img), axis=-1) / 255.0
        else:
            # Shape (1, 6, None, None)
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

            combined_img = np.concatenate((left_img, right_img), axis=-1) / 255.0

        combined_img = combined_img.transpose(2, 0, 1)

        return np.expand_dims(combined_img, 0).astype(np.float32)


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
    use_onnx = False
    if use_onnx:
        # Initialize model
        model_path = '../HITStereo/weights/model_float32_opt.onnx'
        depth_estimator = HITStereo_ONNX(model_path)
    else:
        # Initialize model
        # model_path = './weights/model_float32_optfp16.engine'
        # model_path = './weights/model_float32_optfp32.engine'
        model_path = 'weights/hitnet_xl_sf_finalpass_from_tf_480x640fp16.engine'
        # model_path = './weights/model_float32fp32.engine'
        depth_estimator = HITStereo_TRT(model_path)
    # Load images
  