# -*- coding: utf-8 -*-
# @Time    : 2025/6/24 19:09
# @Author  : sjh
# @Site    : 
# @File    : open3d_sdk.py
# @Comment : 可视化点云

import open3d as o3d
import numpy as np
import queue
import threading
from os import path


def visualize_pointscloud(show_q):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)
    pointcloud = o3d.geometry.PointCloud()
    to_reset = True
    vis.add_geometry(pointcloud)

    while True:
        try:
            pcd = show_q.get()

            pcd = np.asarray(pcd.points).reshape((-1, 3))
            pointcloud.points = o3d.utility.Vector3dVector(pcd)
            # vis.update_geometry()
            # 注意，如果使用的是open3d 0.8.0以后的版本，这句话应该改为下面格式
            vis.update_geometry(pointcloud)
            if to_reset:
                vis.reset_view_point(True)
                to_reset = False
            vis.poll_events()
            vis.update_renderer()
        except:
            continue


if __name__ == '__main__':

    show_q = queue.Queue(1)
    visual = threading.Thread(target=visualize_pointscloud, args=(show_q,))
    visual.start()

    frame = 0
    import time

    while True:
        print(f"Simulating frame {frame}...")

        # 模拟生成 (480, 640, 3) 的点云数据，范围 [-1, 1]
        points = np.random.uniform(-1, 1, size=(480 * 640, 3))

        # 构建 open3d 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if show_q.full():
            show_q.get()
        show_q.put(pcd)

        frame += 1
        frame %= 98  # 可选：循环帧编号打印
        time.sleep(0.03)  # 控制帧率，可调节