import cv2
import numpy as np
import time
from config import Stereo
Stereo = Stereo()
# 视频路径
video_path = r"/calibration/rectified_video2.avi"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 初始化SGBM参数
def create_sgbm():
    window_size = 5
    min_disp = -32
    num_disp = 32 - min_disp  # 必须是16的整数倍
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size**2,  # 视差平滑参数
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    return stereo

sgbm = create_sgbm()

# 时域平滑滤波器
prev_disp = None
alpha = 0.3  # 平滑系数 (0-1)，越小越平滑
frame_id = 0
while cap.isOpened():
    ret, combined_image = cap.read()
    if not ret:
        break
    frame_id += 1
    # 分割左右视图
    rectifyed_left = combined_image[:combined_image.shape[0]//2, :, :]
    rectifyed_right = combined_image[combined_image.shape[0]//2:, :, :]
    rectifyed_left = cv2.resize(rectifyed_left, (640, 480))
    rectifyed_right = cv2.resize(rectifyed_right, (640, 480))
    # 转换为灰度图
    gray_left = cv2.cvtColor(rectifyed_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rectifyed_right, cv2.COLOR_BGR2GRAY)

    # SGBM计算视差
    start_time = time.time()
    for i in range(1):
        disp = sgbm.compute(gray_left, gray_right).astype(np.float32) / 16.0  # SGBM返回的视差需要除以16
        # print(disp.min(), disp.max())

        # 时域平滑 (IIR滤波器)
        if prev_disp is not None:
            disp = alpha * disp + (1 - alpha) * prev_disp
        prev_disp = disp.copy()

        # 后处理
        disp = cv2.medianBlur(disp, 5)  # 中值滤波去噪
        disp = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))  # 闭运算填充空洞
    end_time = time.time()
    print(f"推理时间: {end_time - start_time:.4f} 秒")
    if frame_id < 250:
        continue
    
    Stereo.show_depth_point(disp, rectifyed_left)
    
    # 可视化
    valid_disp = disp[disp > 0]  # 只统计有效视差
    if len(valid_disp) > 0:
        vmin, vmax = np.percentile(valid_disp, [5, 95])  # 动态范围裁剪
        disp_vis = np.clip((disp - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
    else:
        disp_vis = np.zeros_like(disp, dtype=np.uint8)
    
    colored = cv2.applyColorMap(disp_vis, cv2.COLORMAP_PLASMA)
    cv2.imshow("SGBM Disparity", colored)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()