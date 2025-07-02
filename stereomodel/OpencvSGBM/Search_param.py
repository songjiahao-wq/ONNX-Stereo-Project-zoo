# -*- coding: utf-8 -*-
# @Time    : 2025/5/21 21:40
# @Author  : sjh
# @Site    : 
# @File    : Search_param.py
# @Comment :
"""
| 方法           | 优点        | 缺点      |
| ------------ | --------- | ------- |
| Grid Search  | 简单、容易实现   | 组合多时很慢  |
| 有GT指标优化      | 精确        | 需要真实视差图 |
| `optuna` 等工具 | 高效、自动探索最优 | 安装依赖稍麻烦 |

我有多个GT指标的真实视差图，给出自动调参代码

需要我帮你改写为带 Ground Truth 的版本，或支持 batch 图像自动调参也可以继续告诉我！

改进评分函数？

加入 batch 图像平均评分？

加入 Ground Truth 支持？
"""
import cv2
import numpy as np
import itertools
import time

def disparity_score(disparity, gt_disp, mask):
    """
    其中 mask 是有效区域掩码，比如：mask = (gt_disp > 0) & (gt_disp < 192)  # 你根据数据集来设定
    """
    epe = np.abs(disparity - gt_disp)[mask].mean()
    return -epe

def test_sgbm(left_img,right_img,best_params):
    max_disp = best_params['numDisparities']  # 最大视差，一定是16的倍数
    left_img = cv2.resize(left_img, (320 * 2, 240 * 2))
    right_img = cv2.resize(right_img, (320 * 2, 240 * 2))
    # 1. 扩展左右图像左边界（padding）
    # 这里用复制边缘像素扩展，也可以用其他方式扩展\
    left_img_pad = left_img
    right_img_pad = right_img
    # left_img_pad = cv2.copyMakeBorder(left_img, 0, 0, max_disp, 0, cv2.BORDER_REPLICATE)
    # right_img_pad = cv2.copyMakeBorder(right_img, 0, 0, max_disp, 0, cv2.BORDER_REPLICATE)

    # 2. 计算视差（注意计算的是扩展后图像）
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=max_disp,
        blockSize=best_params['blockSize'],
        P1=8 * 3 * 5 * 5,
        P2=32 * 3 * 5 * 5,
        disp12MaxDiff=1,
        uniquenessRatio=best_params['uniquenessRatio'],
        speckleWindowSize=best_params['speckleWindowSize'],
        speckleRange=best_params['speckleRange'],
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    print(left_img_pad.shape)
    t1 = time.time()
    for i in range(1):
        disparity_pad = stereo.compute(left_img_pad, right_img_pad).astype(np.float32) / 16.0
    print(time.time() - t1)
    # 3. 裁剪回原图大小，丢弃扩展的部分
    disparity = disparity_pad[:, max_disp:]
    disp = disparity
    norm = ((disp - disp.min()) / (disp.max() - disp.min()) * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(norm, cv2.COLORMAP_PLASMA)
    # disparity 的宽度现在等于原图宽度，边缘缺失的问题得到缓解
    cv2.imshow("disp Image", depth_colormap)
    cv2.waitKey(0)



def search1():


    def compute_disparity(left, right, params):
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=params['numDisparities'],
            blockSize=params['blockSize'],
            uniquenessRatio=params['uniquenessRatio'],
            speckleWindowSize=params['speckleWindowSize'],
            speckleRange=params['speckleRange'],
            P1=8 * 1 * params['blockSize'] ** 2,
            P2=32 * 1 * params['blockSize'] ** 2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        disparity = stereo.compute(left, right).astype(np.float32) / 16.0
        return disparity

    def score_disparity(disparity):
        # 简单的评分方式：视差图的边缘数量、连续性等
        # 这里只是示意，可以换成更复杂的评分，比如和GT比较
        return -np.std(disparity[disparity > 0])  # 方差越小越平滑

    # 参数空间
    param_grid = {
        'numDisparities': [32, 64, 96],
        'blockSize': [3, 5, 7],
        'uniquenessRatio': [10, 15],
        'speckleWindowSize': [50, 100],
        'speckleRange': [1, 2],
    }

    # 构建所有参数组合
    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())

    # 输入图像
    left_img = cv2.imread(r'D:\BaiduSyncdisk\work\Stereo\ONNX-Stereo-Project-zoo\data\480x640\im1.png', cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(r'D:\BaiduSyncdisk\work\Stereo\ONNX-Stereo-Project-zoo\data\480x640\im0.png', cv2.IMREAD_GRAYSCALE)

    best_score = -np.inf
    best_params = None

    for values in param_combinations:
        params = dict(zip(param_names, values))
        disp = compute_disparity(left_img, right_img, params)
        score = score_disparity(disp)
        print(f"Params: {params}, Score: {score}")
        if score > best_score:
            best_score = score
            best_params = params

    print("Best Parameters:", best_params)
    test_sgbm(left_img, right_img, best_params)
def search2():
    import cv2
    import numpy as np
    import optuna

    # === 1. 输入图像 ===
    left_img = cv2.imread(r'D:\BaiduSyncdisk\work\Stereo\ONNX-Stereo-Project-zoo\data\480x640\im1.png', cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(r'D:\BaiduSyncdisk\work\Stereo\ONNX-Stereo-Project-zoo\data\480x640\im0.png', cv2.IMREAD_GRAYSCALE)

    # 可选：对图像进行边界扩展（如果你使用了这个 trick）
    def pad_image(img, max_disp):
        return cv2.copyMakeBorder(img, 0, 0, max_disp, 0, cv2.BORDER_REPLICATE)

    MAX_DISPARITY = 64
    left_pad = pad_image(left_img, MAX_DISPARITY)
    right_pad = pad_image(right_img, MAX_DISPARITY)

    # === 2. 定义评分函数（没有GT，简单用平滑性评估） ===
    def disparity_score(disparity):
        valid_disp = disparity[disparity > 0]
        if len(valid_disp) < 100:  # 视差太少，跳过
            return float('-inf')
        smoothness = -np.std(valid_disp)  # 方差越小，视差图越平滑
        return smoothness

    # === 3. Optuna目标函数 ===
    def objective(trial):
        # 搜索参数空间
        numDisparities = trial.suggest_categorical("numDisparities", [16, 32, 64, 96, 128])
        blockSize = trial.suggest_int("blockSize", 3, 11, step=2)
        uniquenessRatio = trial.suggest_int("uniquenessRatio", 5, 20)
        speckleWindowSize = trial.suggest_int("speckleWindowSize", 0, 200)
        speckleRange = trial.suggest_int("speckleRange", 1, 32)

        P1 = 8 * 1 * blockSize ** 2
        P2 = 32 * 1 * blockSize ** 2

        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=numDisparities,
            blockSize=blockSize,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            P1=P1,
            P2=P2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        disparity = stereo.compute(left_pad, right_pad).astype(np.float32) / 16.0
        disparity = disparity[:, MAX_DISPARITY:]  # 裁剪回来

        score = disparity_score(disparity)
        return score

    # === 4. 执行调参 ===
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)  # 可以改为更多次，比如100

    # === 5. 输出最优参数 ===
    print("Best Score:", study.best_value)
    print("Best Params:", study.best_params)
    test_sgbm(left_img, right_img, study.best_params)
def search3():
    import cv2
    import numpy as np
    import optuna

    # === 1. 加载图像和 Ground Truth 视差 ===
    left = cv2.imread('../../data/480x640/im1.png', cv2.IMREAD_GRAYSCALE)
    right = cv2.imread('../../data/480x640/im0.png', cv2.IMREAD_GRAYSCALE)
    gt_disp = cv2.imread('../../data/480x640/480x640dispgt.png', cv2.IMREAD_UNCHANGED).astype(np.float32) / 16.0  # GT 视差

    # 必要时边缘扩展（防止左侧无值）
    MAX_DISP = 128
    left_pad = cv2.copyMakeBorder(left, 0, 0, MAX_DISP, 0, cv2.BORDER_REPLICATE)
    right_pad = cv2.copyMakeBorder(right, 0, 0, MAX_DISP, 0, cv2.BORDER_REPLICATE)

    # === 2. Ground Truth 的有效 mask（避免空值、遮挡）===
    gt_mask = (gt_disp > 0) & (gt_disp < MAX_DISP)  # 可调

    # === 3. 目标函数 ===
    def objective(trial):
        numDisparities = trial.suggest_categorical("numDisparities", [64, 96, 128])
        blockSize = trial.suggest_int("blockSize", 3, 11, step=2)
        uniquenessRatio = trial.suggest_int("uniquenessRatio", 5, 20)
        speckleWindowSize = trial.suggest_int("speckleWindowSize", 0, 200)
        speckleRange = trial.suggest_int("speckleRange", 1, 32)

        P1 = 8 * 1 * blockSize ** 2
        P2 = 32 * 1 * blockSize ** 2

        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=numDisparities,
            blockSize=blockSize,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            P1=P1,
            P2=P2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        disparity = stereo.compute(left_pad, right_pad).astype(np.float32) / 16.0
        disparity = disparity[:, MAX_DISP:]  # 裁剪回原图尺寸

        # === 计算 EPE（仅在有效区域）===
        mask = gt_mask & (disparity > 0)
        if np.sum(mask) < 100:
            return float('inf')  # 无效视差图，给一个大误差

        epe = np.abs(disparity[mask] - gt_disp[mask]).mean()
        return epe  # Optuna 默认最小化目标

    # === 4. 启动搜索 ===
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    # === 5. 输出最优结果 ===
    print("Best EPE:", study.best_value)
    print("Best Parameters:", study.best_params)
    test_sgbm(left_pad, right_pad, study.best_params)
if __name__ == '__main__':
    search1()