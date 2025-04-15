# -*- coding: utf-8 -*-
# @Time    : 2025/3/20 15:47
# @Author  : sjh
# @File    : convert.py
# @Comment : Convert ONNX to TensorRT engine (FP16/FP32)

import tensorrt as trt
import os
from packaging import version
trt_version = trt.__version__
compare_version = version.parse("8.6.1")

# 比较版本
if version.parse(trt_version) > compare_version:
    trt_version8_bool = False
    print(f"✅ TensorRT 版本 {trt_version} 大于或等于 {compare_version}")
else:
    trt_version8_bool = True
    print(f"❌ TensorRT 版本 {trt_version} 小于 {compare_version}")

def convert_onnx_to_trt(onnx_path, fp16=True):
    """
    将 ONNX 转换为 TensorRT Engine
    :param onnx_path: ONNX 文件路径
    :param fp16: 是否开启 FP16（默认开启）
    """
    # TensorRT 日志
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # 创建 Builder 和 Network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 读取 ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print(f"❌ 解析 {onnx_path} 失败！错误信息如下：")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return

    # 创建 Builder 配置
    config = builder.create_builder_config()

    # 是否开启 FP16
    if fp16 and builder.platform_has_fast_fp16:
        print("✅ 开启 FP16 模式")
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        print("⚠️ 设备不支持 FP16，使用 FP32")

    # 创建动态输入 Profile
    profile = builder.create_optimization_profile()
    min_shape = (1, 3, 392, 528)  # 最小输入
    opt_shape = (1, 3, 392, 528)  # 最佳输入
    max_shape = (1, 3, 392, 528)  # 最大输入

    profile.set_shape("left", min_shape, opt_shape, max_shape)
    profile.set_shape("right", min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    # 生成 Engine
    if trt_version8_bool:
        engine = builder.build_engine(network, config)
    else:
        engine = builder.build_serialized_network(network, config)
    if engine is None:
        print(f"❌ 生成 TensorRT engine 失败！")
        return

    # 保存 Engine
    if fp16:
        engine_path = onnx_path.replace(".onnx", "fp16.engine")
    else:
        engine_path = onnx_path.replace(".onnx", "fp32.engine")
    with open(engine_path, "wb") as f:
        if trt_version8_bool:
            f.write(engine.serialize())
        else:
            f.write(engine)

    print(f"✅ 成功生成 TensorRT engine: {engine_path} 🚀")


# 运行转换
onnx_models = ["stereomodel\HITStereo\weights\model_float32_opt.onnx"]
# onnx_models = ["stereomodel\HITStereo\weights\model_float32_opt.onnx"]
# onnx_models = ["stereomodel/IGEV++RTStereo/weights/rt_sceneflow_640480.onnx"]
# onnx_models = ["stereomodel/AANet/weights/gmstereo-scale1-sceneflow-124a438f_1x3x480x640_sim.onnx"]
# onnx_models = ["stereomodel/FastAVCnet/weights/fast_acvnet_sceneflow_opset16_480x640.onnx"]
for model in onnx_models:
    if os.path.exists(model):
        convert_onnx_to_trt(model, fp16=True)
    else:
        print(f"❌ ONNX 模型 {model} 不存在，跳过转换！")
