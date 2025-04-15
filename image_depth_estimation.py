import cv2
import numpy as np
from imread_from_url import imread_from_url

from crestereo import CREStereo
#深度模型预处理，后处理函数
# 预处理函数
height,width= 352,640
def preprocess_image(image):
    image = cv2.resize(image, (width, height))
    image_input = np.transpose(image, (2, 0, 1))[None, :, :, :].astype(np.float32)
    return np.ascontiguousarray(image_input)
import cv2
import open3d as o3d
from matplotlib import pyplot as plt
depth_cam_matrix = np.array([[238.494, 0, 229.981],
                             [0, 233.194, 151.422],
                             [0, 0, 1]])

# depth_cam_matrix = np.array([[455.61, 0, 325.9],
#                              [0,  461.22, 234.91],
#                              [0, 0, 1]])
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
    global depth_map, depth_cam_matrix1
    rectifyed_left = cv2.resize(rectifyed_left, (width, height))
    depth_map = disparity_to_depth(disp1, depth_cam_matrix[0, 0], 70.9062)

    max_depth = 10  # 单位 m
    depth_scale = 1000  # 假设深度图以 mm 存储
    depth_map = np.where(depth_map / depth_scale > max_depth, 0, depth_map)
    depth_map = np.where(depth_map / depth_scale < 0, 0, depth_map)
    while True:

        depth_colormap = visualize_depth(disp1)
        rectifyed_left = cv2.resize(rectifyed_left, (depth_colormap.shape[1], depth_colormap.shape[0]))
        print(rectifyed_left.shape,depth_colormap.shape)
        combined_image = np.hstack((rectifyed_left, depth_colormap))

        cv2.imshow("Estimated depth222", combined_image)
        cv2.setMouseCallback("Estimated depth222", on_mouse, 0)
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
# Model Selection options (not all options supported together)
iters = 5            # Lower iterations are faster, but will lower detail. 
		             # Options: 2, 5, 10, 20 

shape = (320, 640)   # Input resolution.
				     # Options: (240,320), (320,480), (380, 480), (360, 640), (480,640), (720, 1280)

version = "combined" # The combined version does 2 passes, one to get an initial estimation and a second one to refine it.
					 # Options: "init", "combined"

# Initialize model
model_path = f'models/resources_iter2/crestereo_init_iter2_360x640.onnx'
depth_estimator = CREStereo(model_path)

# Load images
# left_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im2.png")
# right_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im6.png")
left_img = cv2.imread(r"D:\BaiduSyncdisk\work\Stereo\data\test2\im0.png")
right_img = cv2.imread(r"D:\BaiduSyncdisk\work\Stereo\data\test2\im1.png")
# Estimate the depth
disparity_map = depth_estimator(left_img, right_img)
show_depth_point(disparity_map, left_img)
color_disparity = depth_estimator.draw_disparity()
combined_image = np.hstack((left_img, color_disparity))

cv2.imwrite("out.jpg", combined_image)

cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)	
cv2.imshow("Estimated disparity", combined_image)
cv2.waitKey(0)

cv2.destroyAllWindows()
