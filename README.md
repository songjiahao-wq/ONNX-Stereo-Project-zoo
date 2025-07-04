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

| 模型名                | 简要说明                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------- |
| **AANet**          | Adaptive Aggregation Network，提出可学习的区域聚合模块，在准确性和效率之间取得良好平衡。适用于端到端的立体匹配任务。              |
| **CGIStereo**      | 融合上下文引导和交叉注意力机制，增强特征交互能力，提高细节区域和边界的匹配效果。                                              |
| **COEXStereo**     | Cooperative Experts Stereo，结合多个专家分支协同处理不同结构区域，提升立体匹配的鲁棒性。                             |
| **CRESTereo**      | 基于递归上下文聚合与自适应残差优化机制的立体网络，兼顾全局感知和局部细节重建。                                               |
| **FastAVCNet**     | Fast Attention Volume Completion Network，结合注意力机制与体积补全策略，专为实时应用优化，速度快、资源消耗低。           |
| **HITStereo**      | Hierarchical Interaction Transformer Stereo，引入层次化Transformer交互结构，有效增强左右视图之间的全局依赖建模能力。 |
| **IGEV**           | Iterative Geometry Encoding Volume，提出迭代几何体积表示结构，通过多尺度几何编码提升匹配精度。                      |
| **IGEV++RTStereo** | 实时优化版本的 IGEV，融合轻量设计与稀疏体积推理，兼顾速度与精度，适用于部署场景。                                           |
| **IGEVPLUS**       | IGEV 的改进版本，加入更深层次的上下文整合和增强监督机制，以进一步提升模型鲁棒性和泛化能力。                                      |
| **OpencvSGBM**     | OpenCV 中基于 Semi-Global Matching 的传统立体算法，计算效率高，适用于无深度学习部署需求的嵌入式设备。                     |
| **RAFTStereo**     | 基于光流网络 RAFT 的 Stereo 版本，采用基于递归更新的匹配机制，精度极高，在 KITTI 等基准上表现优异。                          |
| **Tihit**          | 通常为自定义或调试用模型目录，可包含测试模型或用于功能验证的简单结构（需根据实际代码内容判断）。                                      |

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