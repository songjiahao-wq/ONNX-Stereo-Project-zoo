import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time
cuda.init()
trt.init_libnvinfer_plugins(None, "")
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
is_trt10 = int(trt.__version__.split(".")[0]) >= 10


class trt_infer_time:
    def __init__(self, engine_file_path):
        self.engine_file_path = engine_file_path
        self.engine = self.load_engine(engine_file_path)
        self.context = None
        self.stream = None
        self.d_inputs = []
        self.d_outputs = []

        # 获取引擎信息
        if is_trt10:
            self.num_bindings = self.engine.num_io_tensors
            self.input_names = [self.engine.get_tensor_name(i) for i in range(self.num_bindings)
                                if
                                self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
            self.output_names = [self.engine.get_tensor_name(i) for i in range(self.num_bindings)
                                 if
                                 self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
            self.num_inputs = len(self.input_names)
        else:
            self.num_bindings = self.engine.num_bindings
            self.num_inputs = sum([self.engine.binding_is_input(i) for i in range(self.num_bindings)])
            self.input_names = [self.engine.get_binding_name(i) for i in range(self.num_inputs)]
            self.output_names = [self.engine.get_binding_name(i) for i in range(self.num_inputs, self.num_bindings)]

    def __del__(self):
        # 确保在对象销毁时释放所有资源
        self.release_resources()

    def release_resources(self):
        # 释放CUDA内存
        for d_mem in self.d_inputs + self.d_outputs:
            if d_mem:
                d_mem.free()

        # 重置列表
        self.d_inputs = []
        self.d_outputs = []

        # 释放上下文
        if self.context:
            del self.context
            self.context = None

        # 释放流
        if self.stream:
            self.stream = None

    def load_engine(self, engine_file_path):
        with open(engine_file_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            return runtime.deserialize_cuda_engine(f.read())

    def prepare_inference(self, batch_size=1):
        # 创建执行上下文
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # 准备输入数据
        input_shapes = []
        input_data_list = []

        if is_trt10:
            for name in self.input_names:
                shape = list(self.engine.get_tensor_shape(name))
                shape = [batch_size if dim == -1 else dim for dim in shape]
                input_shapes.append(shape)
                input_data = np.random.rand(*shape).astype(np.float32)
                input_data_list.append(input_data)
                self.context.set_input_shape(name, shape)
        else:
            for i in range(self.num_inputs):
                shape = list(self.engine.get_binding_shape(i))
                shape = [batch_size if dim == -1 else dim for dim in shape]
                input_shapes.append(shape)
                input_data = np.random.rand(*shape).astype(np.float32)
                input_data_list.append(input_data)
                self.context.set_binding_shape(i, shape)

        # 分配输入内存
        self.d_inputs = []
        self.h_inputs = []
        for data in input_data_list:
            h_input = np.ascontiguousarray(data)
            d_input = cuda.mem_alloc(h_input.nbytes)
            cuda.memcpy_htod(d_input, h_input)
            self.h_inputs.append(h_input)
            self.d_inputs.append(d_input)

        # 分配输出内存
        self.d_outputs = []
        self.h_outputs = []
        if is_trt10:
            for name in self.output_names:
                shape = list(self.context.get_tensor_shape(name))
                shape = [batch_size if dim == -1 else dim for dim in shape]
                h_output = np.empty(shape, dtype=np.float32)
                d_output = cuda.mem_alloc(h_output.nbytes)
                self.h_outputs.append(h_output)
                self.d_outputs.append(d_output)
        else:
            for i in range(self.num_inputs, self.num_bindings):
                shape = list(self.context.get_binding_shape(i))
                shape = [batch_size if dim == -1 else dim for dim in shape]
                h_output = np.empty(shape, dtype=np.float32)
                d_output = cuda.mem_alloc(h_output.nbytes)
                self.h_outputs.append(h_output)
                self.d_outputs.append(d_output)

        # 设置TensorRT 10+的输入输出地址
        if is_trt10:
            for name, d_input in zip(self.input_names, self.d_inputs):
                self.context.set_tensor_address(name, int(d_input))
            for name, d_output in zip(self.output_names, self.d_outputs):
                self.context.set_tensor_address(name, int(d_output))

    def measure_speed(self, iterations=1000):
        self.prepare_inference()

        # 预热
        for _ in range(10):
            if is_trt10:
                self.context.execute_async_v3(self.stream.handle)
            else:
                bindings = [int(d) for d in self.d_inputs + self.d_outputs]
                self.context.execute_async_v2(bindings, self.stream.handle)
            self.stream.synchronize()

        # 正式测量
        start_time = time.time()
        for _ in range(iterations):
            if is_trt10:
                self.context.execute_async_v3(self.stream.handle)
            else:
                bindings = [int(d) for d in self.d_inputs + self.d_outputs]
                self.context.execute_async_v2(bindings, self.stream.handle)
            self.stream.synchronize()
        end_time = time.time()

        avg_time = (end_time - start_time) * 1000 / iterations
        print(f"Average inference time over {iterations} iterations: {avg_time:.2f} ms")

        # 释放资源
        self.release_resources()
        return avg_time


# 使用示例
trt_infer = trt_infer_time('stereomodel/HITStereo/weights/model_float32_optfp16.engine')
trt_infer.measure_speed(iterations=1000)
