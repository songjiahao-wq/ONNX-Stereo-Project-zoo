import numpy as np
import json
import open3d as o3d


def read_calib(calib_path):
    # 读取标定文件
    with open(calib_path, 'r') as f:
        calib_data = json.load(f)

    left_k = np.array(calib_data['left_camera_intrinsics']['camera_matrix'])
    right_k = np.array(calib_data['right_camera_intrinsics']['camera_matrix'])
    left_distortion = np.array(calib_data['left_camera_intrinsics']['distortion_coefficients'])
    right_distortion = np.array(calib_data['right_camera_intrinsics']['distortion_coefficients'])
    # left_Q = np.array(calib_data['left_camera_intrinsics']['Q_left'])
    # right_Q = np.array(calib_data['right_camera_intrinsics']['Q_right'])
    rotation_matrix = np.array(calib_data['rotation_matrix'])
    translation_vector = np.array(calib_data['translation_vector'])
    Q_matrix = np.array(calib_data['Q_matrix'])
    # inverse_extrinsic = np.array(calib_data['extrinsic']['inverse_extrinsic_matrix'])
    # R = inverse_extrinsic[:3, :3]
    # T = inverse_extrinsic[:3, 3]
    # T = T.reshape(3, 1)
    # T = np.array(calib_data['extrinsic']['t'])

    return left_k, right_k, left_distortion, right_distortion, rotation_matrix, translation_vector, Q_matrix


calib_path = r'D:\BaiduSyncdisk\work\Stereo\ONNX-Stereo-Project-zoo\utils\cali_circle2.json'
left_k, right_k, left_distortion, right_distortion, r, t, Q_matrix = read_calib(calib_path)


def get_disparity(min_depth, Q):
    focal_length = Q[2, 3]
    focal_length = focal_length.astype(float)
    base_line_inverse = Q[3, 2]
    base_line_inverse = base_line_inverse.astype(float)
    pixel_dis = focal_length / float(min_depth * base_line_inverse)
    return pixel_dis


min_depth, max_depth = [1, 10]
max_disparity = abs(get_disparity(min_depth, Q_matrix))
min_disparity = abs(get_disparity(max_depth, Q_matrix))


def init_des(w, h, factor):
    arr = np.zeros((h, w, 2), dtype=float)
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # 将网格的坐标赋值给 arr
    arr[..., 0] = x * factor
    arr[..., 1] = y * factor
    return arr


def get_points_3d(displ, Q = Q_matrix, min_disparity=max_disparity , max_disparity=min_disparity, factor=2.):
    displ = displ * factor
    des_init = init_des(displ.shape[1], displ.shape[0], factor)
    displ = np.expand_dims(displ, axis=-1)
    xyd = np.concatenate((des_init, displ), axis=-1)
    arr1 = np.ones((displ.shape[0], displ.shape[1], 1))
    xyd1 = np.concatenate((xyd, arr1), axis=-1)
    xyd1_reshaped = xyd1.reshape(-1, 4)
    xyd1_filtered = xyd1_reshaped[xyd1_reshaped[:, 2] > min_disparity]
    xyd1_filtered = xyd1_filtered[xyd1_filtered[:, 2] < max_disparity]

    XYZW = np.dot(Q, xyd1_filtered.T)
    # XYZW = XYZW.T
    XYZ = XYZW[:3, :] / XYZW[[3], :]
    # XYZ_world = np.dot(rl, XYZ) + tl
    # points_3d = XYZ_world.T

    return XYZ.T


class pick_cloud():
    def pick_points222(self, pcd):
        print("Please pick points using [shift + left click]")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # 用户可以选择点
        vis.destroy_window()
        return vis.get_picked_points()

    def pick_points(self, pcd, reset=False):
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        if reset:
            vis.clear_geometries()  # Clears any previous geometries
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()
        picked_indices = vis.get_picked_points()
        # E 检索拾取点的坐标
        picked_points = np.asarray(pcd.points)[picked_indices]
        return picked_points


def draw_points3d(points3d, target_points=None):
    # 筛选出有效点
    mask = np.isfinite(points3d).all(axis=1)
    valid_points = points3d[mask]
    print(f"Valid points shape: {valid_points.shape}")

    # 创建点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(valid_points)

    main = pick_cloud()

    # 计算当前点云的点数
    num_points = len(valid_points)
    print(f"Original number of points: {num_points}")

    # 如果点数超过目标点数，则进行体素下采样
    if target_points and num_points > target_points:
        # 估算合适的体素大小，比例关系是目标点数与原始点数的比例
        voxel_size = np.cbrt(num_points / target_points) * 0.01  # 乘以0.1来微调体素大小
        print(f"Estimated voxel size: {voxel_size}")

        # 执行体素下采样
        downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size)
        print(f"Downsampled point cloud has {len(np.asarray(downsampled_point_cloud.points))} points.")

        # # 使用下采样后的点云进行展示
        # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # o3d.visualization.draw_geometries([downsampled_point_cloud, axis])

        picked_indices = main.pick_points(downsampled_point_cloud, reset=True)
    else:
        # # 如果点数已经小于目标点数，则直接展示
        # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # o3d.visualization.draw_geometries([point_cloud, axis])

        picked_indices = main.pick_points(point_cloud, reset=True)
        # # 检测按键输入
        # key = cv2.waitKey(1) & 0xFF  # 获取键盘按键的 ASCII 码
        # # 如果按下 'q' 键，退出循环
        # if key == ord('q'):
        #     print("Exiting loop...")
        #     break
