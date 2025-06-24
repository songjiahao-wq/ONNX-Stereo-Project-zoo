# -*- coding: utf-8 -*-
# @Time    : 2025/6/24 19:17
# @Author  : sjh
# @Site    : 
# @File    : 111.py
# @Comment : 可视化点云
import pyvista as pv
import numpy as np

points = np.random.rand(1000, 3)
cloud = pv.PolyData(points)
plotter = pv.Plotter()
plotter.add_points(cloud, render_points_as_spheres=True)
plotter.show()