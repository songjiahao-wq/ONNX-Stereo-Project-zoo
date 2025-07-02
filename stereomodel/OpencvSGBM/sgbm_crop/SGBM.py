import cv2
import numpy as np
import time
cv2.setNumThreads(4)  # 设置最大线程
from scipy.ndimage import distance_transform_edt
from stereomodel.OpencvSGBM.utils.filterz_disp import fill_depth_nearest_ROI

def apply_disparity_filter(disparity_map, left_image=None,right_image=None, filter_type='wls', stereo=None, **kwargs):
    """
    对视差图应用不同的滤波器
    :param disparity_map: 输入的视差图（CV_32F类型）
    :param left_image: 左图像（用于需要引导图像的方法）
    :param filter_type: 滤波器类型 ('none', 'median', 'gaussian', 'bilateral', 'guided', 'wls', 'fgs')
    :param kwargs: 滤波器参数
    :return: 滤波后的视差图
    """
    # 确保视差图是浮点型
    disparity_map = disparity_map.astype(np.float32)

    # 有效视差区域掩码（排除无效值）
    valid_mask = (disparity_map > kwargs.get('min_disp', 0)) & \
                 (disparity_map < kwargs.get('max_disp', 255))

    if filter_type == 'none':
        filtered = disparity_map.copy()

    elif filter_type == 'median':
        # 中值滤波（去除孤立噪声点）
        ksize = kwargs.get('ksize', 5)
        filtered = cv2.medianBlur(disparity_map, ksize)

    elif filter_type == 'gaussian':
        # 高斯滤波（平滑处理）
        ksize = kwargs.get('ksize', (5, 5))
        sigma = kwargs.get('sigma', 1.5)
        filtered = cv2.GaussianBlur(disparity_map, ksize, sigma)

    elif filter_type == 'bilateral':
        # 双边滤波（保持边缘）
        d = kwargs.get('d', 9)
        sigma_color = kwargs.get('sigma_color', 75)
        sigma_space = kwargs.get('sigma_space', 75)
        filtered = cv2.bilateralFilter(disparity_map, d, sigma_color, sigma_space)

    elif filter_type == 'guided':
        # 导向滤波（需要引导图像）
        if left_image is None:
            raise ValueError("Guided filter requires left image")

        # 转换为灰度图
        guide = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)

        # 归一化
        disp_norm = cv2.normalize(disparity_map, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        guide_norm = cv2.normalize(guide, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # 应用导向滤波
        radius = kwargs.get('radius', 5)
        eps = kwargs.get('eps', 0.01)
        guided_filter = cv2.ximgproc.createGuidedFilter(guide_norm, radius, eps)
        filtered = guided_filter.filter(disp_norm)

        # 恢复原始范围
        filtered = filtered * (disparity_map.max() - disparity_map.min()) + disparity_map.min()

    elif filter_type == 'wls':
        # WLS滤波（加权最小二乘滤波）
        if left_image is None or right_image is None:
            raise ValueError("WLS filter requires both left and right images")
        print("Input Disparity - Min: {}, Max: {}, Mean: {}".format(np.min(disparity_map), np.max(disparity_map), np.mean(disparity_map)))

        # 计算左图视差图
        # disparity_map = stereo.compute(left_image, right_image).astype(np.float32) / 16.0
        print("Recalculated Disparity - Min: {}, Max: {}, Mean: {}".format(np.min(disparity_map), np.max(disparity_map), np.mean(disparity_map)))

        # 创建右视差计算器（WLS需要）
        right_matcher = cv2.ximgproc.createRightMatcher(stereo)

        # 计算右图视差图
        disparity_right = right_matcher.compute(left_image, right_image).astype(np.float32) / 16.0

        # 创建WLS滤波器并设置参数
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
        wls_filter.setLambda(80000)  # 强度参数，控制平滑程度
        wls_filter.setSigmaColor(1.5)  # 颜色空间平滑参数

        # 应用WLS滤波器
        filtered = wls_filter.filter(disparity_map, left_image, None, disparity_right)
        filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)
        print(np.max(filtered))
    elif filter_type == 'fgs':
        # Fast Global Smoother（快速全局平滑）
        if left_image is None:
            raise ValueError("FGS filter requires left image")

        # 转换为灰度图
        guide = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)

        # 归一化
        disp_norm = cv2.normalize(disparity_map, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        guide_norm = cv2.normalize(guide, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # 创建FGS滤波器
        fgs_filter = cv2.ximgproc.createFastGlobalSmootherFilter(
            guide_norm,
            lambda_=kwargs.get('lambda_', 1000),
            sigma=kwargs.get('sigma', 1.5)
        )

        # 应用FGS滤波
        filtered = fgs_filter.filter(disp_norm)

        # 恢复原始范围
        filtered = filtered * (disparity_map.max() - disparity_map.min()) + disparity_map.min()

    else:
        filtered = disparity_map.copy()

    # 保持无效区域不变
    filtered = np.where(valid_mask, filtered, disparity_map)
    return filtered
# 滤波器选项
filters = {
    '0': 'none',
    '1': 'median',
    '2': 'gaussian',
    '3': 'bilateral',
    '4': 'guided',
    '5': 'wls',
    '6': 'fgs'
}
current_filter = filters['0']  # 默认使用WLS滤波
class SGBM:
    def __init__(self, use_blur=True):
        self.prev_disp = None
        self.alpha = 0.3  # 平滑系数 (0-1)，越小越平滑
        self.use_blur = use_blur
        self.sgbm = self.create_sgbm()
        self.num_disp = 32  # 视差范围
    def create_sgbm(self):
        window_size = 5
        min_disp = 0
        num_disp = self.num_disp = 32  # 必须是16的整数倍
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
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY# STEREO_SGBM_MODE_SGBM_3WAY ,STEREO_SGBM_MODE_HH, STEREO_SGBM_MODE_SGBM, STEREO_SGBM_MODE_HH4,STEREO_SGBM_MODE_HH4的速度最快，STEREO_SGBM_MODE_HH的精度最好
        )
        return stereo
    def create_bm_matcher(self):
        stereo = cv2.StereoBM_create(
            numDisparities=64,   # 视差范围（必须是16的倍数）
            blockSize=15         # 匹配块大小（奇数，建议5~25）
        )
        return stereo

    def estimate_depth(self, left_image, right_image, down_process=False, copyMake=True):
        """
        进行深度估计推理，返回视差图（深度图）。
        """
        if copyMake:
            left_image = cv2.copyMakeBorder(left_image, 0, 0, self.num_disp, 0, cv2.BORDER_REPLICATE)
            right_image = cv2.copyMakeBorder(right_image, 0, 0, self.num_disp, 0, cv2.BORDER_REPLICATE)
        # 转换为灰度图
        if down_process:
            left_image = cv2.resize(left_image, (
            left_image.shape[1] // 2, left_image.shape[0] // 2), interpolation=cv2.INTER_AREA)
            right_image = cv2.resize(right_image, (
            right_image.shape[1] // 2, right_image.shape[0] // 2), interpolation=cv2.INTER_AREA)

        gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # gray_left = clahe.apply(gray_left)
        # gray_right = clahe.apply(gray_right)
        # gray_left = cv2.equalizeHist(gray_left)
        # gray_right = cv2.equalizeHist(gray_right)

        disp = self.sgbm.compute(gray_left, gray_right).astype(np.float32) / 16.0  # SGBM返回的视差需要除以16
        if down_process:
            disp = self.upscale_disparity(disp, scale_x=2, scale_y=2)  # → 640x480
        if copyMake:
            disp = disp[:, 64:]

            # # 应用选择的滤波器
            # if current_filter in ['guided', 'wls', 'fgs']:
            #     disp = apply_disparity_filter(
            #         disp, left_image, current_filter,
            #         lambda_=8000, sigma=1.5
            #     )
            # else:
            #     disp = apply_disparity_filter(disp, filter_type=current_filter)

        # if self.prev_disp is not None:
        #     disp = self.alpha * disp + (1 - self.alpha) * self.prev_disp
        # self.prev_disp = disp.copy()
        # disp = cv2.medianBlur(disp, 5)  # 中值滤波
        disp = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, np.ones((8, 8), np.float32))  # 闭运算填充空洞
        return disp
if __name__ == "__main__":
    # Stereo = Stereo()
    sgbm = SGBM()#.create_sgbm_test()
    # 视频路径
    video_path = r"/home/orangepi/sjh/SGBM_port_sjh/a0_mjpg.avi"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        exit()
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


        # SGBM计算视差
        start_time = time.time()
        for i in range(1):
            # gray_left = cv2.cvtColor(rectifyed_left, cv2.COLOR_BGR2GRAY)
            # gray_right = cv2.cvtColor(rectifyed_right, cv2.COLOR_BGR2GRAY)
            # disp = sgbm.compute(gray_left, gray_right).astype(np.float32) / 16.0  # SGBM返回的视差需要除以16
            # # print(disp.min(), disp.max())

            # # 时域平滑 (IIR滤波器)
            # if prev_disp is not None:
            #     disp = alpha * disp + (1 - alpha) * prev_disp
            # prev_disp = disp.copy()

            # # 后处理
            # # disp = cv2.medianBlur(disp, 5)  # 中值滤波去噪
            # disp = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))  # 闭运算填充空洞
            disp = sgbm.estimate_depth(rectifyed_left, rectifyed_right)
        end_time = time.time()
        print(f"推理时间: {end_time - start_time:.4f} 秒")
        print(f"视差范围: {disp.min()} - {disp.max()}")
    #     if frame_id < 1:
    #         continue

    #     Stereo.show_depth_point(disp, rectifyed_left)

    #     # 可视化
    #     valid_disp = disp[disp > 0]  # 只统计有效视差
    #     if len(valid_disp) > 0:
    #         vmin, vmax = np.percentile(valid_disp, [5, 95])  # 动态范围裁剪
    #         disp_vis = np.clip((disp - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
    #     else:
    #         disp_vis = np.zeros_like(disp, dtype=np.uint8)

    #     colored = cv2.applyColorMap(disp_vis, cv2.COLORMAP_PLASMA)
    #     cv2.imshow("SGBM Disparity", colored)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()