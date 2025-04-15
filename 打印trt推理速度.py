import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time

trt.init_libnvinfer_plugins(None, "")
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
is_trt10 = int(trt.__version__.split(".")[0]) >= 10  # is TensorRT >= 10


class trt_infer_time:
    def __init__(self, engine_file_path):
        self.cuda_ctx = pycuda.autoinit.context
        self.engine_file_path = engine_file_path
        self.engine = self.load_engine(engine_file_path)

        self.num_bindings = None
        self.input_names = None
        self.num_inputs = None

    def measure_speed(self, ):
        # 加载引擎
        engine = self.load_engine(self.engine_file_path)

        # 获取输入张量的数量和形状
        if is_trt10:
            self.num_bindings = engine.num_io_tensors
            self.input_names = [engine.get_tensor_name(i) for i in range(self.num_bindings) if
                                engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
            self.num_inputs = len(self.input_names)  # Get the actual number of input tensors
        else:
            self.num_bindings = engine.num_bindings
            self.num_inputs = sum([engine.binding_is_input(i) for i in range(self.num_bindings)])

        # 为推理创建随机输入数据 (注意确保输入数据符合引擎的输入尺寸要求)
        input_data_list = []
        batch_size = 1  # 设置具体的批次大小
        print(f"Number of inputs: {self.num_inputs}")

        if is_trt10:
            for name in self.input_names:  # Iterate through input names instead of range
                input_shape = engine.get_tensor_shape(name)  # 获取每个输入的形状

                # 替换动态批次大小 -1 为具体的 batch_size
                input_shape = tuple([batch_size if dim == -1 else dim for dim in input_shape])
                print(f"Input {name} shape after replacing dynamic dimensions: {input_shape}")

                input_data = np.random.rand(*input_shape).astype(np.float32)
                input_data_list.append(input_data)
        else:
            # Original code for non-TRT10 case
            for i in range(self.num_bindings):
                if engine.binding_is_input(i):
                    input_shape = tuple(engine.get_binding_shape(i))
                    input_shape = tuple([batch_size if dim == -1 else dim for dim in input_shape])
                    print(f"Input {i} shape after replacing dynamic dimensions: {input_shape}")
                    input_data = np.random.rand(*input_shape).astype(np.float32)
                    input_data_list.append(input_data)
        # 测试推理速度
        self.cuda_ctx.push()

        output = self.infer_and_measure_speed(engine, input_data_list, iterations=1000)
        self.cuda_ctx.pop()

    # 加载 TensorRT 引擎
    def load_engine(self, engine_file_path):
        with open(engine_file_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    # 进行推理并测量推理时间
    def infer_and_measure_speed(self, engine, input_data_list, iterations=1000):
        context = engine.create_execution_context()
        # 获取输入和输出的数量
        if is_trt10:
            num_inputs = self.num_inputs
            num_outputs = self.num_bindings - num_inputs
            # 获取输入和输出的名称
            input_names = [self.engine.get_tensor_name(i) for i in range(self.num_bindings)
                           if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
            output_names = [self.engine.get_tensor_name(i) for i in range(self.num_bindings)
                            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]

            # 设置每个输入的形状
            for name, input_data in zip(input_names, input_data_list):
                context.set_input_shape(name, input_data.shape)
        else:
            num_inputs = sum([self.engine.binding_is_input(i) for i in range(self.num_bindings)])
            num_outputs = self.num_bindings - num_inputs
            input_names = [self.engine.get_binding_name(i) for i in range(num_inputs)]
            output_names = [self.engine.get_binding_name(i) for i in range(num_inputs, self.num_bindings)]

            # 设置每个输入的形状
            for i, input_data in enumerate(input_data_list):
                context.set_binding_shape(i, input_data.shape)

        # 分配输入内存
        h_inputs = [np.ascontiguousarray(input_data) for input_data in input_data_list]
        d_inputs = [cuda.mem_alloc(h_input.nbytes) for h_input in h_inputs]

        # 分配输出内存
        h_outputs = []
        d_outputs = []
        if is_trt10:
            for name in output_names:
                output_shape = context.get_tensor_shape(name)
                if output_shape[0] == -1:
                    output_shape[0] = 1
                h_output = np.empty(output_shape, dtype=np.float32)
                h_outputs.append(h_output)
                d_output = cuda.mem_alloc(h_output.nbytes)
                d_outputs.append(d_output)
        else:
            for i in range(num_inputs, self.num_bindings):
                output_shape = context.get_binding_shape(i)
                if output_shape[0] == -1:
                    output_shape[0] = 1
                h_output = np.empty(output_shape, dtype=np.float32)
                h_outputs.append(h_output)
                d_output = cuda.mem_alloc(h_output.nbytes)
                d_outputs.append(d_output)

        # 创建 CUDA 流
        stream = cuda.Stream()

        # TensorRT 10+ 特定设置
        if is_trt10:
            # 设置输入张量地址
            for name, d_input in zip(input_names, d_inputs):
                context.set_tensor_address(name, int(d_input))

            # 设置输出张量地址
            for name, d_output in zip(output_names, d_outputs):
                context.set_tensor_address(name, int(d_output))
        else:
            # 传统绑定方式
            bindings = [int(d_input) for d_input in d_inputs] + [int(d_output) for d_output in d_outputs]

        # 测量推理时间
        start_time = time.time()
        for _ in range(iterations):
            # 将输入数据异步传输到设备
            for d_input, h_input in zip(d_inputs, h_inputs):
                cuda.memcpy_htod_async(d_input, h_input, stream)

            # 执行推理
            if is_trt10:
                context.execute_async_v3(stream_handle=stream.handle)
            else:
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

            # 从设备异步拷贝输出数据到主机
            for h_output, d_output in zip(h_outputs, d_outputs):
                cuda.memcpy_dtoh_async(h_output, d_output, stream)

            # 确保数据传输和推理同步完成
            stream.synchronize()
        end_time = time.time()

        # 计算平均推理时间
        total_time = end_time - start_time
        avg_inference_time = total_time / iterations
        print(f"Average inference time over {iterations} iterations: {avg_inference_time * 1000:.2f} ms")
        return h_outputs


# 运行速度测量函数
trt_infer_time_instance = trt_infer_time(engine_file_path=r'stereomodel\IGEV\weights\IGEV_middlebury_HxWfp16.engine')
trt_infer_time_instance.measure_speed()
# measure_speed(engine_file=r"D:\BaiduSyncdisk\work\RTMpose-26-port\rtmpose-mobilev4\combined_model2.engine")
# measure_speed(engine_file='./weights/end2end.engine')
# measure_speed(engine_file='./weights/new_pose.engine')
