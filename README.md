# ONNX Stereo 项目集合

欢迎来到 **ONNX Stereo 项目集合**，这是一个为实时性能优化的立体匹配算法集合，使用了 ONNX 格式。该仓库集成了多个流行的深度学习立体匹配模型，包括 **HitNet**、**IGEV**、**FastAVCNet** 等，并针对立体深度估计任务进行了优化。

---

## 🚀 功能

- **ONNX 优化**：所有模型均经过 ONNX 格式优化，支持高效推理。
- **多种模型**：包含 **HitNet**、**IGEV**、**FastAVCNet** 等先进的立体匹配算法。
- **高性能**：实现实时的立体深度估计，推理速度快。
- **跨平台支持**：支持 CPU 和 GPU（CUDA）加速，利用 ONNX Runtime 和 TensorRT 进行推理。
- **模块化设计**：轻松切换和测试不同的模型，或者以最小的修改集成新的模型。

---

## 📂 包含的模型

- **HitNet**：一种快速且准确的立体匹配模型，使用分层特征匹配方法。
- **IGEV**：一种高效的深度学习立体匹配算法，利用图像梯度和嵌入体积进行处理。
- **FastAVCNet**：一种快速且轻量级的模型，针对实时立体深度估计进行了优化。

---

## 🛠️ 安装

### 前提条件

1. **Python 3.x**（推荐：Python 3.6+）
2. **PyTorch**（用于训练和测试模型）
3. **ONNX** 和 **ONNX Runtime**（用于优化推理）
4. **CUDA**（如果你计划使用 GPU）

### 安装步骤

克隆此仓库：

```bash
git clone https://github.com/yourusername/ONNX-Stereo-Project-zoo.git
cd ONNX-Stereo-Project-zoo
```
## 一 calib
1. 首先需要拍摄一段标定视频,视频保存方式为左a0.avi,右a1.avi

2. 然后进行标定:
```bash
python calibration/stereoRectify_process.py --img_path ./data/left.png --img_path2 ./data/right.png --method 2
```
3. 开始矫正视频

保存的视频在路径父目录下
```bash
python calibration/stereoRectify_process.py --method 3
```

## onnxsim
 **method 1**
```bash
onnxsim export_models/raftstereo-middlebury_dynamic.onnx ^
      export_models/raftstereo-middlebury_dynamic_optimized.onnx
```
```bash
onnxsim export_models/raftstereo-middlebury_static.onnx ^
      export_models/raftstereo-middlebury_static_optimized.onnx ^
      --input-shape "left_image:1,3,384,512" "right_image:1,3,384,512"
```
 **method 2**
```bash
python utils/onnx_sim.py
```
## onnxsimer

## reference:
- [ONNX](https://onnx.ai/)
- [IGEV]()