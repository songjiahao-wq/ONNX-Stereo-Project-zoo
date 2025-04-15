import cv2
import numpy as np
from imread_from_url import imread_from_url

from crestereo import CREStereo

# 深度模型预处理，后处理函数


import cv2
import open3d as o3d
import json
from matplotlib import pyplot as plt


class Stereo:
    def __init__(self):

        # 预处理函数
        height, width = 784, 1056
        left_k, right_k, left_distortion, right_distortion, r, t, q = self.read_calib("20250305133700.json")
        r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(left_k, left_distortion, right_k, right_distortion,
                                                          (width, height), r, t, alpha=0)

        self.map_1x, self.map_1y = cv2.initUndistortRectifyMap(left_k, left_distortion, r1, p1, (width, height), cv2.CV_32FC1)
        self.map_2x, self.map_2y = cv2.initUndistortRectifyMap(right_k, right_distortion, r2, p2, (width, height), cv2.CV_32FC1)

        self.fx, self.fy, self.cx, self.cy = p1[0, 0], p1[1, 1], p1[0, 2], p1[1, 2]
        self.baseline = abs(1 / q[3, 2]) if q[3, 2] != 0 else 0
        self.depth_cam_matrix = np.array([[self.fx, 0, self.cx],
                                          [0, self.fy, self.cy],
                                          [0, 0, 1]])
        self.focal_length = self.depth_cam_matrix[0, 0]  # 768.80165
        self.baseline = 70.9062  # 7cm
        
        
        self.depth_map = None
        self.scale = 1
    def reset_calib(self, scale):
        self.fx, self.fy = self.fx * scale, self.fy * scale
        self.depth_cam_matrix = np.array([[self.fx, 0, self.cx],
                                          [0, self.fy, self.cy],
                                          [0, 0, 1]])
        
    def read_calib(self, calib_path):
        # 读取标定文件
        with open(calib_path, 'r') as f:
            calib_data = json.load(f)

        left_k = np.array(calib_data['left_camera_intrinsics']['camera_matrix'])
        right_k = np.array(calib_data['right_camera_intrinsics']['camera_matrix'])
        left_distortion = np.array(calib_data['left_camera_intrinsics']['distortion_coefficients'])
        right_distortion = np.array(calib_data['right_camera_intrinsics']['distortion_coefficients'])
        rotation_matrix = np.array(calib_data['rotation_matrix'])
        translation_vector = np.array(calib_data['translation_vector'])
        Q_matrix = np.array(calib_data['Q_matrix'])

        return left_k, right_k, left_distortion, right_distortion, rotation_matrix, translation_vector, Q_matrix

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

    def visualize_depth(self, depth_filtered, colormap=cv2.COLORMAP_JET):
        # 归一化到 0-255
        depth_min = np.min(depth_filtered)
        depth_max = np.max(depth_filtered)
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

    def xy_3d(self, x, y, depth_map, depth_cam_matrix, depth_scale=1000):
        """ 将图像坐标 (x, y) 转换为 3D 世界坐标 """
        fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
        cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]

        z = self.depth_map[y, x] / depth_scale  # 获取像素点的深度值 (m)
        if z <= 0:
            return np.array([None, None, None])  # 如果深度无效，返回空

        X = (x - cx) * z / fx
        Y = (y - cy) * z / fy

        return np.array([X, Y, z])

    def disparity_to_depth(self, disp, focal_length, baseline):
        """ 视差图转换为深度图 """
        depth = (focal_length * baseline) / (disp + 1e-6)  # 避免除零错误
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
    def rectify_image(self, left_img, right_img, scale=1):
        rectifyed_left = cv2.remap(left_img, self.map_1x, self.map_1y, cv2.INTER_LINEAR)
        rectifyed_right = cv2.remap(right_img, self.map_2x, self.map_2y, cv2.INTER_LINEAR)
        return rectifyed_left, rectifyed_right
    def show_depth_point(self, disp1, rectifyed_left,scale):
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

            cv2.imshow("Estimated depth222", combined_image)
            cv2.setMouseCallback("Estimated depth222", self.on_mouse, 0)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

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


show_depth_point = Stereo()
# Initialize model
model_path = f'models/resources_iter2/crestereo_init_iter2_480x640.onnx'
depth_estimator = CREStereo(model_path)

# 读取视频
video_path = r"D:\BaiduSyncdisk\work\Stereo\data\stereo_video_rectified.avi"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("无法打开视频文件")
    exit()
cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)

# 逐帧读取并显示
frame_id = 1
while cap.isOpened():
    ret, combined_image = cap.read()
    frame_id +=1
    # if frame_id < 10:
    #     continue
    rectifyed_left = combined_image[:640, :, :]
    rectifyed_right = combined_image[640:, :, :]
    # 原始图像尺寸
    # original_h, original_w = rectifyed_left.shape[:2]
    # resize_h, resize_w = 480, 640
    # scale = resize_w / original_w
    # show_depth_point.reset_calib(scale)
    # # rectifyed_left, rectifyed_right = show_depth_point.rectify_image(rectifyed_left, rectifyed_right)
    # # image_width, image_height = [528, 392]
    # factor = float(1280 / resize_w)
    # rectifyed_left = cv2.resize(rectifyed_left, (resize_w, resize_h))
    # rectifyed_right = cv2.resize(rectifyed_right, (resize_w, resize_h))
    # Estimate the depth
    disparity_map = depth_estimator(rectifyed_left, rectifyed_right)
    
    # show_depth_point.show_depth_point(disparity_map, rectifyed_left ,1)
    color_disparity = depth_estimator.draw_disparity()
    combined_image = np.hstack((rectifyed_left, color_disparity))

    # cv2.imwrite("out.jpg", combined_image)

    cv2.imshow("Estimated disparity", combined_image)
    cv2.waitKey(1)

cv2.destroyAllWindows()
