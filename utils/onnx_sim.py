# -*- coding: utf-8 -*-
# @Time    : 2025/4/19 23:53
# @Author  : sjh
# @Site    : 
# @File    : onnx_sim.py
# @Comment :
import onnx
from onnxsim import simplify

def optimize_onnx_model(input_path, output_path, input_shapes=None):
    """
    优化ONNX模型
    Args:
        input_path: 输入模型路径
        output_path: 输出模型路径
        input_shapes: 可选,指定输入形状字典
    """
    # 加载ONNX模型
    model = onnx.load(input_path)

    # 设置输入形状(如果提供)
    if input_shapes:
        model_simp, check = simplify(model, input_shapes=input_shapes)
    else:
        model_simp, check = simplify(model)

    # 检查优化结果
    if check:
        print(f"模型优化成功: {input_path} -> {output_path}")
        onnx.save(model_simp, output_path)
    else:
        print("模型优化失败!")
        return False

    return True
if __name__ == '__main__':
    # 使用示例
    input_shapes = {
        'left_image': [1, 3, 384, 512],
        'right_image': [1, 3, 384, 512]
    }

    optimize_onnx_model(
        'export_models/raftstereo-middlebury_static.onnx',
        'export_models/raftstereo-middlebury_static_optimized.onnx',
        input_shapes
    )
