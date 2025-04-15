import os

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import tensorrt as trt
from packaging import version
trt_version = trt.__version__
compare_version = version.parse("8.6.2")
# 比较版本
if version.parse(trt_version) > compare_version:
    trt_version8_bool = False
    print(f"✅ TensorRT 版本 {trt_version} 大于或等于 {compare_version}")
else:
    trt_version8_bool = True
    print(f"❌ TensorRT 版本 {trt_version} 小于 {compare_version}")
trt.init_libnvinfer_plugins(None, "")

import pycuda.autoinit
import pycuda.driver as cuda
from typing import List, Optional, Tuple, Union
import warnings
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2

height, width = 480, 640


class BaseTRTInference:
    """深度估计模型类"""

    def __init__(self, engine_file):
        # 初始化TensorRT引擎
        self.cuda_ctx = pycuda.autoinit.context
        self.cuda = cuda
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_file, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        # 创建执行上下文
        self.context = self.engine.create_execution_context()

        # ⚠️ **动态 batch 处理：显式设置 batch_size**
        # self.batch_size = 1  # 可以改成任意有效 batch_size
        # if trt_version8_bool:
        #     self.context.set_binding_shape(0, (self.batch_size, 3, height, width))
        #     self.context.set_binding_shape(1, (self.batch_size, 3, height, width))
        # else:
        #     self.context.set_input_shape("input_name_0", (self.batch_size, 3, height, width))
        #     self.context.set_input_shape("input_name_1", (self.batch_size, 3, height, width))

        # 绑定输入输出
        self.inputs = []
        self.outputs = []
        self.bindings = []
        for binding_index in range(self.engine.num_io_tensors):
            binding_name = self.engine.get_tensor_name(binding_index)  # ⚠️ 修正方法
            dtype = self.engine.get_tensor_dtype(binding_name)  # ⚠️ 修正方法
            shape = self.context.get_tensor_shape(binding_name)  # ⚠️ 直接获取实际 shape
            size = trt.volume(shape) * np.dtype(np.float32).itemsize

            print(f"🔹 {binding_name} Shape: {shape}")  # 调试输出
            shape[0] = 1
            host_mem = np.empty(shape, dtype=np.float32)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            if self.engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})
            self.bindings.append(int(device_mem))
        self.height, self.width = height, width
    def __call__(self, left_img, right_img):
        return self.update(left_img, right_img)

    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height))
        image_input = np.transpose(image, (2, 0, 1))[None, :, :, :].astype(np.float32)
        return np.ascontiguousarray(image_input)

    def update(self, left_image, right_image):
        """
        进行深度估计推理，返回视差图（深度图）。
        """
        self.cuda_ctx.push()

        self.img_height, self.img_width = left_image.shape[:2]
        # **1️⃣ 预处理**
        left_image_input = self.preprocess_image(left_image)
        right_image_input = self.preprocess_image(right_image)
        outputs = self.inference(left_image_input, right_image_input)
        # **6️⃣ 解析输出**
        self.disparity_map = self.process_output(outputs)
        self.cuda_ctx.pop()

        return self.disparity_map

    def process_output(self, outputs):
        
        disp_pred = outputs.reshape(1, 1, self.height, self.width)  # 确保 shape 正确
        disp_pred = np.squeeze(disp_pred[:, 0, :, :])  # 移除 batch 维度
        return disp_pred

    def inference(self, left_image_input, right_image_input):
        """
        进行深度估计推理，返回视差图（深度图）。
        """
        # **2️⃣ 传输数据到 GPU**
        cuda.memcpy_htod(self.inputs[0]["device"], left_image_input)
        cuda.memcpy_htod(self.inputs[1]["device"], right_image_input)
        # **3️⃣ 执行 TensorRT 推理**
        success = self.context.execute_v2(bindings=self.bindings)
        # **4️⃣ 从 GPU 拷贝输出**
        
        for i in range(len(self.outputs)):
            cuda.memcpy_dtoh(self.outputs[i]["host"], self.outputs[i]["device"])

        # **5️⃣ 确保输出有效**
        if np.isnan(self.outputs[0]["host"]).sum() > 0:
            raise RuntimeError("[ERROR] 推理输出包含 NaN！")
        # **6️⃣ 解析输出**
        outputs = self.outputs[-1]["host"]

        return outputs

    def draw_disparity(self):
        disparity_map = cv2.resize(self.disparity_map, (self.img_width, self.img_height))
        
        # print(np.max(disparity_map), np.min(disparity_map))
        
        norm_disparity_map = 255 * ((disparity_map - np.min(disparity_map)) /
                                    (np.max(disparity_map) - np.min(disparity_map)))

        return cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map, 1), cv2.COLORMAP_JET)
    def measure_speed(self):
        cuda.Context.synchronize()
        start_time = time.time()
        input_left = np.ascontiguousarray(np.zeros((1, 3, self.height, self.width), dtype=np.float32))
        input_right = np.ascontiguousarray(np.zeros((1, 3, self.height, self.width), dtype=np.float32))
        for _ in range(1000):
            self.inference(input_left, input_right)
        cuda.Context.synchronize()
        end_time = time.time()
        
        print(f"🔹 推理耗时: {(end_time - start_time)} ms)