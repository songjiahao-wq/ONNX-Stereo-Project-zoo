import sys
import argparse
import glob
import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import os
import onnxruntime as ort
import cv2
import open3d as o3d
DEVICE = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def visualize_disparity(disparity_map, title):
    plt.figure(figsize=(10, 6))
    plt.imshow(disparity_map, cmap='jet')
    plt.colorbar(label="Disparity")
    plt.title(title)
    plt.axis('off')
    plt.show()

depth_cam_matrix = np.array([[238.494, 0, 229.981],
                             [0, 233.194, 151.422],
                             [0, 0, 1]])
focal_length = depth_cam_matrix[0, 0]  # 768.80165
baseline = 70.9062  # 7cm

def save_depth(disp):
    # 将视差图转换为深度图，避免除零问题
    epsilon = 1e-6
    # 计算深度图（单位为米），防止除零
    depth = (focal_length * baseline) / (disp + epsilon)

    # 显示归一化后的深度图（仅用于展示，转换为 8 位）
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = np.squeeze(depth_normalized.astype(np.uint8))
    plt.imshow(depth_uint8, cmap="jet")
    plt.show()

    # 保存原始深度图（转换为毫米后保存为 16 位 PNG）
    # 这里假设 depth 单位为米，乘以 1000 转为毫米
    depth_mm = np.squeeze(depth)
    depth_16bit = depth_mm.astype(np.uint16)
    cv2.imwrite('./runs/depth_16bit.png', depth_16bit)
def visualize_depth(depth_filtered, colormap=cv2.COLORMAP_JET):
    # 归一化到 0-255
    depth_min = np.min(depth_filtered)
    depth_max = np.max(depth_filtered)
    depth_norm = (depth_filtered - depth_min) / (depth_max - depth_min)  # 归一化到 0-1
    depth_vis = (depth_norm * 255).astype(np.uint8)  # 转换为 0-255 范围

    # 伪彩色映射
    depth_colormap = cv2.applyColorMap(depth_vis, colormap)
    return depth_colormap
def on_mouse(event, x, y, flags, param):
    """ 鼠标点击事件，获取像素点的 3D 坐标 """
    global depth_map, depth_cam_matrix

    if event == cv2.EVENT_LBUTTONDOWN:
        point_3d = xy_3d(x, y, depth_map, depth_cam_matrix)
        if None in point_3d:
            print(f"点 ({x}, {y}) 的深度无效")
        else:
            print(f"点 ({x}, {y}) 的三维坐标: X={point_3d[0]:.3f}, Y={point_3d[1]:.3f}, Z={point_3d[2]:.3f} m")
def xy_3d(x, y, depth_map, depth_cam_matrix, depth_scale=1000):
    """ 将图像坐标 (x, y) 转换为 3D 世界坐标 """
    fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
    cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]

    z = depth_map[y, x] / depth_scale  # 获取像素点的深度值 (m)
    if z <= 0:
        return np.array([None, None, None])  # 如果深度无效，返回空

    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy

    return np.array([X, Y, z])
def disparity_to_depth(disp, focal_length, baseline):
    """ 视差图转换为深度图 """
    depth = (focal_length * baseline) / (disp + 1e-6)  # 避免除零错误
    return depth
def depth2xyz(depth_map, depth_cam_matrix, flatten=False, depth_scale=1000):
    """ 将深度图转换为点云 """
    fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
    cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]

    # 生成网格
    h, w = np.mgrid[0:depth_map.shape[0], 0:depth_map.shape[1]]

    # 计算 3D 坐标
    z = depth_map / depth_scale  # 归一化深度（单位: m）
    x = (w - cx) * z / fx
    y = (h - cy) * z / fy
    xyz = np.dstack((x, y, z)) if flatten == False else np.dstack((x, y, z)).reshape(-1, 3)

    return xyz
def show_depth_point(disp1, rectifyed_left):
    global depth_map, depth_cam_matrix

    depth_map = disparity_to_depth(disp1, depth_cam_matrix[0, 0], 70.9062)

    max_depth = 7  # 单位 m
    depth_scale = 1000  # 假设深度图以 mm 存储
    depth_map = np.where(depth_map / depth_scale > max_depth, 0, depth_map)
    depth_map = np.where(depth_map / depth_scale < 0, 0, depth_map)
    while True:

        depth_colormap = visualize_depth(disp1)
        rectified_left = cv2.resize(rectifyed_left, (disp1.shape[1], disp1.shape[0]))
        combined_image = np.hstack((rectified_left, depth_colormap))

        cv2.imshow("Estimated depth111", combined_image)
        cv2.setMouseCallback("Estimated depth111", on_mouse, 0)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

        pc = depth2xyz(depth_map, depth_cam_matrix, flatten=True)

        # **可视化点云并允许选点**
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="Click to Get Depth")
        vis.add_geometry(pcd)

        print("\n请在窗口中 **按住Shift + 左键** 选点，然后关闭窗口后查看选中的点。\n")

        vis.run()  # 运行可视化（允许选点）
        vis.destroy_window()

        # **获取选中的点**
        picked_points = vis.get_picked_points()
        if picked_points:
            print("\n选中的 3D 点坐标：")
            for i, idx in enumerate(picked_points):
                x, y, z = pc[idx]
                print(f"点 {i + 1}: X={x:.3f}, Y={y:.3f}, Z={z:.3f} m")
        else:
            print("未选中任何点。")
def demo(args):


    # 导出 ONNX 模型
    onnx_model_path = args.output_onnx
    imfile1 = args.left_imgs
    imfile2 = args.right_imgs
    image1 = load_image(imfile1)
    image2 = load_image(imfile2)
    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)

    # ONNX 推理
    ort_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    input_l_np = image1.cpu().numpy()
    input_r_np = image2.cpu().numpy()
    onnx_inputs = {"left": input_l_np, "right": input_r_np}
    onnx_outputs = ort_session.run(None, onnx_inputs)
    disp_onnx = onnx_outputs[0].squeeze()
    show_depth_point(disp_onnx, cv2.imread(r'D:\BaiduSyncdisk\work\Stereo\ONNX-Stereo-Project-zoo\calibration\rectified_right copy.png'))
    print(disp_onnx.shape)
    visualize_disparity(disp_onnx, title="ONNX Disparity Map")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames",
                        default=r"D:\BaiduSyncdisk\work\Stereo\ONNX-Stereo-Project-zoo\calibration\rectified_left copy.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames",
                        default=r"D:\BaiduSyncdisk\work\Stereo\ONNX-Stereo-Project-zoo\calibration\rectified_right copy.png")
    parser.add_argument('--output_directory', help="directory to save output",
                        default="./demo_output")
    parser.add_argument('--mixed_precision', action='store_true', default=True, help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'],
                        help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--valid_iters', type=int, default=18, help='number of flow-field updates during forward pass')
    parser.add_argument('--output_onnx', help="path to save the ONNX model", 
                        default=r"D:\BaiduSyncdisk\work\Stereo\IGEV-plusplus-YOLO8\IGEVplusplus\pretrained\igev_ONNX\rt_model_simplified.onnx")


    args = parser.parse_args()
    demo(args)
