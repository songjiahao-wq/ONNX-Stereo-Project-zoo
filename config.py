# -*- coding: utf-8 -*-
# @Time    : 2025/3/24 下午4:08
# @Author  : sjh
# @Site    : 
# @File    : config.py
# @Comment :
import numpy as np
import cv2
import json
import open3d as o3d
from matplotlib import pyplot as plt
calib_data = {
    "stereo0": {
        "cam0": {
            "cam_overlaps": [1],
            "camera_model": "pinhole",
            "distortion_coeffs": [0.75575636, 0.09272825, -0.00010397, -2.675e-05, 0.00094747, 1.13111697, 0.27240028,
                                  0.01046174],
            "distortion_model": "radtan",
            "intrinsics": [788.4632328500001, 788.70885008, 961.0316543599999, 527.5127149800001],  # fx, fy, cx, cy
            "resolution": [1920, 1080]
        },
        "cam1": {
            "T_cn_cnm1": [
                [0.99997798, 0.00066776, -0.00660303, -0.07090623],
                [-0.00073086, 0.99995405, -0.00955833, 0.0],
                [0.00659635, 0.00956294, 0.99993252, 0.0]
            ],
            "camera_model": "pinhole",
            "distortion_coeffs": [0.75575636, 0.09272825, 0.00013049, -1.3e-07, 0.00094747, 1.13111697, 0.27240028,
                                  0.01046174],
            "distortion_model": "radtan",
            "intrinsics": [786.9761893, 786.9329483400001, 962.2218096, 521.79998306],  # fx, fy, cx, cy
            "resolution": [1920, 1080]
        }
    }
}
class CameraIntrinsics:
    def getIntrinsics_AI(self):

        height = 1056
        width = 784
        p = [
            752.598, 0.0, 535.039, 33.82135389646828,
            0.0, 752.598, 386.481, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        # p = [
        #     751.366, 0.0, 537.485, 33.82135389646828,
        #     0.0, 752.904, 383.444, 0.0,
        #     0.0, 0.0, 1.0, 0.0
        # ]
        # p = [
        #     745.59004729, 0.0, 439.3305969238281, 33.82135389646828,
        #     0.0, 745.59004729, 391.6967468261719, 0.0,
        #     0.0, 0.0, 1.0, 0.0
        # ]
        baseline = 0.10689529687646554
        return height, width, p, baseline
    def getIntrinsics1920_1080(self):
        height = 1080
        width = 1920
        [f, cx, cy, baseline]=[715.481, 689.943, 464.59, 0.0709062]
        p = [
            788.4632328, 0.0, 961.03165436, 33.82135389646828,
            0.0, 788.70885008, 527.51271498, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        # 635.982 715.47975 719.8649368286133 464.5906677246094 70.90622931718826
        # p = [
        #     635.982, 0.0, 719.8649368286133, 0,
        #     0.0, 715.47975, 464.5906677246094, 0.0,
        #     0.0, 0.0, 1.0, 0.0
        # ]
        baseline = 0.07090622931718826
        return height, width, p, baseline
    def getIntrinsics1280_640(self):
        height = 640
        width = 1280

        p = [
            476.987060546875, 0.0, 459.9617614746094, 33.82135389646828,
            0.0, 423.9884948730469, 275.3126220703125, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        # 1280*640计算好的
        p = [
            423.988, 0.0, 479.9099578857422, 33.82135389646828,
            0.0, 423.988, 275.31298828125, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        # p = [
        #     460.3950380000503, 0.0, 620.1596069335938, 33.82135389646828,
        #     0.0, 460.3950380000503, 267.4479331970215, 0.0,
        #     0.0, 0.0, 1.0, 0.0
        # ]
        baseline = 0.07090622931718826
        return height, width, p, baseline
    def getIntrinsics640_480(self):
        height = 480
        width = 640
        [fx, fy, cx, cy, baseline] = [252.562, 252.562, 309.352, 147.295, 70.0427]

        p = [
            fx, 0.0, cx, 33.82135389646828,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        # 矫正后
        fx = 229.98088073730466
        fy = 229.98088073730466
        cx = 329.4670867919922
        cy = 206.48446655273438
        fx = 229.98088073730466
        fy = 229.98088073730466
        cx = 318.0451965332031
        cy = 206.48446655273438
        p = [
            fx, 0.0, cx, 33.82135389646828,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        # p = [
        #     460.3950380000503, 0.0, 620.1596069335938, 33.82135389646828,
        #     0.0, 460.3950380000503, 267.4479331970215, 0.0,
        #     0.0, 0.0, 1.0, 0.0
        # ]
        baseline = 0.0700427
        return height, width, p, baseline
    def getIntrinsics640_352(self):
        height = 352
        width = 640
        p = [
            238.4935302734375, 0.0, 233.19366455078125, 33.82135389646828,
            0.0, 229.9808807373047, 151.42193603515625, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        baseline = 0.07090622931718826
        return height, width, p, baseline


class Stereo:
    def __init__(self, res_height=640, res_width=1280):
        ori_height, ori_width, p, baseline = CameraIntrinsics().getIntrinsics_AI()

        self.fx, self.fy, self.cx, self.cy, self.baseline = p[0], p[5], p[2], p[6], baseline * 1000
        scale_x = res_width / ori_width  # 宽度缩放比例
        scale_y = res_height / ori_height  # 高度缩放比例
        self.reset_calib(scale_x, scale_y)  # 调整内参
        self.depth_cam_matrix = np.array([[self.fx, 0, self.cx],
                                          [0, self.fy, self.cy],
                                          [0, 0, 1]])
        self.focal_length = self.depth_cam_matrix[0, 0]  # 768.80165

        self.depth_map = None
        print(self.fx, self.fy, self.cx, self.cy, self.baseline)

    def reset_calib(self, scale_x, scale_y):
        self.fx, self.fy, self.cx, self.cy = self.fx * scale_x, self.fy * scale_y, self.cx * scale_x, self.cy * scale_y
        self.depth_cam_matrix = np.array([[self.fx, 0, self.cx],
                                          [0, self.fy, self.cy],
                                          [0, 0, 1]])

    def filter_depth(self, depth_map):
        max_depth = 10  # 单位 m
        depth_scale = 1000  # 假设深度图以 mm 存储
        depth_map = np.where(depth_map / depth_scale > max_depth, 0, depth_map)
        depth_map = np.where(depth_map / depth_scale < 0, 0, depth_map)
        # depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)
        return depth_map

    def save_depth(self, disp):
        # 将视差图转换为深度图，避免除零问题
        epsilon = 1e-6
        # 计算深度图（单位为米），防止除零
        depth = (self.focal_length * self.baseline) / (disp + epsilon)

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

    def visualize_disp(self, depth_filtered, colormap=cv2.COLORMAP_MAGMA):
        # 归一化到 0-255
        depth_min = 0.3376  # np.min(depth_filtered)
        depth_max = 20.0000  # np.max(depth_filtered)
        depth_norm = (depth_filtered - depth_min) / (depth_max - depth_min)  # 归一化到 0-1
        depth_vis = (depth_norm * 255).astype(np.uint8)  # 转换为 0-255 范围

        # 伪彩色映射
        depth_colormap = cv2.applyColorMap(depth_vis, colormap)
        return depth_colormap

    def on_mouse(self, event, x, y, flags, param):
        """ 鼠标点击事件，获取像素点的 3D 坐标 """

        if event == cv2.EVENT_LBUTTONDOWN:

            point_3d = self.xy_3d(x, y, self.depth_map, self.depth_cam_matrix)
            if None in point_3d:
                print(f"点 ({x}, {y}) 的深度无效")
            else:
                print(f"点 ({x}, {y}) 的三维坐标: X={point_3d[0]:.3f}, Y={point_3d[1]:.3f}, Z={point_3d[2]:.3f} m")

    def xy_3d(self, x, y, depth_map=None, depth_cam_matrix=None, depth_scale=1000):
        """ 将图像坐标 (x, y) 转换为 3D 世界坐标 """
        fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
        cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]

        z = depth_map[y, x] / depth_scale  # 获取像素点的深度值 (m)
        if z <= 0:
            return np.array([None, None, None])  # 如果深度无效，返回空

        X = (x - cx) * z / fx
        Y = (y - cy) * z / fy

        return np.array([X, Y, z])

    def disparity_to_depth(self, disp, focal_length, baseline):
        """ 视差图转换为深度图 """
        depth = (focal_length * baseline) / disp
        return depth

    def depth2xyz(self, depth_map, depth_cam_matrix, flatten=False, depth_scale=1000):
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

    def rectify_image(self, left_img, right_img, ):
        rectifyed_left = cv2.remap(left_img, self.map_1x, self.map_1y, cv2.INTER_LINEAR)
        rectifyed_right = cv2.remap(right_img, self.map_2x, self.map_2y, cv2.INTER_LINEAR)
        return rectifyed_left, rectifyed_right

    def disp_combine(self, disp1, rectifyed_left):
        if self.depth_map is None:
            self.depth_map = self.disparity_to_depth(disp1,self.fx,self.baseline)
        depth_colormap = self.visualize_disp(disp1)
        rectifyed_left = cv2.resize(rectifyed_left, (depth_colormap.shape[1], depth_colormap.shape[0]))
        combined_image = np.hstack((rectifyed_left, depth_colormap))
        cv2.imshow("Estimated disparity", combined_image)
        cv2.waitKey(1)

    def visualize_depth(self, depth_filtered, colormap=cv2.COLORMAP_JET):
        # 归一化到 0-255
        depth_min = np.min(depth_filtered)
        depth_max = np.max(depth_filtered)
        depth_norm = (depth_filtered - depth_min) / (depth_max - depth_min)  # 归一化到 0-1
        depth_vis = (depth_norm * 255).astype(np.uint8)  # 转换为 0-255 范围

        # 伪彩色映射
        depth_colormap = cv2.applyColorMap(depth_vis, colormap)
        return depth_colormap
    def show_depth_point(self, disp1, rectifyed_left, scale=1):
        self.scale = scale

        rectifyed_left = cv2.resize(rectifyed_left, (640, 480))
        self.depth_map = self.disparity_to_depth(disp1, self.fx, self.baseline)

        max_depth = 10  # 单位 m
        depth_scale = 1000  # 假设深度图以 mm 存储
        self.depth_map = np.where(self.depth_map / depth_scale > max_depth, 0, self.depth_map)
        self.depth_map = np.where(self.depth_map / depth_scale < 0, 0, self.depth_map)
        while True:
            depth_colormap = self.visualize_depth(disp1)
            rectifyed_left = cv2.resize(rectifyed_left, (depth_colormap.shape[1], depth_colormap.shape[0]))
            print(rectifyed_left.shape, depth_colormap.shape)
            combined_image = np.hstack((rectifyed_left, depth_colormap))
            cv2.imshow("Estimated disparity222", combined_image)
            cv2.setMouseCallback("Estimated disparity222", self.on_mouse, 0)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
            show_ply = True
            if show_ply:
                pc = self.depth2xyz(self.depth_map, self.depth_cam_matrix, flatten=True)

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


if __name__ == '__main__':
    CameraIntrinsics = Stereo()
    # print(CameraIntrinsics.getIntrinsics1280_640())