# -*- coding: utf-8 -*-
# @Time    : 2025/3/29 15:52
# @Author  : sjh
# @Site    : 
# @File    : 立体校正与绘制.py
# @Comment :
# -*- coding: utf-8 -*-
# @Time    : 2025/3/29 15:15
# @Author  : sjh
# @Site    :
# @File    : 111.py
# @Comment :
import cv2
import numpy as np
from matplotlib import pyplot as plt


class StereoRectify:
    def detect_and_match(self, left_img, right_img):
        """特征检测与匹配"""
        # 初始化ORB检测器
        orb = cv2.ORB_create(nfeatures=1000)

        # 检测特征点和计算描述子
        kp1, des1 = orb.detectAndCompute(left_img, None)
        kp2, des2 = orb.detectAndCompute(right_img, None)

        # 使用BFMatcher进行匹配
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
    
        # 按距离排序并选择最佳匹配
        matches = sorted(matches, key=lambda x: x.distance)[:100]

        # 提取匹配点坐标
        left_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        right_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        return left_pts, right_pts, matches
    def detect_and_match2(self, left_img, right_img):
        """特征检测与匹配"""
        # 初始化ORB检测器
        orb = cv2.ORB_create(nfeatures=1000)

        # 检测特征点和计算描述子
        kp1, des1 = orb.detectAndCompute(left_img, None)
        kp2, des2 = orb.detectAndCompute(right_img, None)

        # 使用BFMatcher进行匹配
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
    
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(left_img, None)
        kp2, des2 = sift.detectAndCompute(right_img, None)
        flann = cv2.FlannBasedMatcher()
        matches = flann.knnMatch(des1, des2, k=2)
        # 应用比率测试筛选优质匹配
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:  # Lowe's ratio test
                good_matches.append(m)
        
        # 按距离排序并选择前100个最佳匹配
        good_matches = sorted(good_matches, key=lambda x: x.distance)[:100]
        
        # 提取匹配点坐标
        left_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        right_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    
        return left_pts, right_pts, good_matches
    def analyze_disparity(self, left_pts, right_pts):
        """视差分析"""
        disparities = left_pts - right_pts
        y_diffs = disparities[:, 1]

        # 计算统计指标
        metrics = {
            "mean_y": np.mean(np.abs(y_diffs)),
            "std_y": np.std(y_diffs),
            "valid_ratio": np.sum(np.abs(y_diffs) < 1.0) / len(y_diffs),
            "disparities": disparities
        }
        return metrics

    def draw_matches(self, left_img, right_img, left_pts, right_pts):
        """绘制匹配点"""
        vis_img = np.hstack((left_img, right_img))

        if len(vis_img.shape) == 2:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)

        # 绘制连接线
        for i in range(len(left_pts)):
            pt1 = tuple(map(int, left_pts[i]))
            pt2 = tuple(map(int, right_pts[i] + [left_img.shape[1], 0]))
            cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)
            cv2.circle(vis_img, pt1, 3, (0, 0, 255), -1)
            cv2.circle(vis_img, pt2, 3, (0, 0, 255), -1)

        return vis_img

    def draw_metrics(self, image, metrics):
        """在图像上绘制统计指标"""
        text = f"AvgY: {metrics['mean_y']:.2f}px  StdY: {metrics['std_y']:.2f}px  Valid: {metrics['valid_ratio'] * 100:.1f}%"
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return image

    def show_disparity_histogram(self, disparities):
        """显示视差直方图"""
        plt.figure(figsize=(10, 4))

        plt.subplot(121)
        plt.hist(disparities[:, 0], bins=50, color='blue')
        plt.title('Horizontal Disparity')
        plt.xlabel('Pixels')
        plt.ylabel('Count')

        plt.subplot(122)
        plt.hist(disparities[:, 1], bins=50, color='red')
        plt.title('Vertical Disparity')
        plt.xlabel('Pixels')
        plt.ylabel('Count')

        plt.tight_layout()
        plt.pause(0.01)

    def rectify_video(self, rect_left , rect_right):
        print(rect_left.shape)
        print(rect_right.shape)

        # 转换为灰度图用于特征检测
        gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

        # 特征检测与匹配
        left_pts, right_pts, matches = self.detect_and_match(gray_left, gray_right)

        # 视差分析
        metrics = self.analyze_disparity(left_pts, right_pts)

        # 可视化组件
        match_vis = self.draw_matches(rect_left, rect_right, left_pts, right_pts)
        metric_vis = self.draw_metrics(match_vis.copy(), metrics)

        # 显示结果
        cv2.imshow('Stereo Analysis', cv2.resize(metric_vis, (1920, 1080)))
        self.show_disparity_histogram(metrics["disparities"])

        cv2.waitKey(0)

        # cv2.destroyAllWindows()
        # plt.close()

StereoRectify = StereoRectify()
flag = 2
if flag == 1:
    img_left = cv2.imread(r"D:\BaiduSyncdisk\work\Stereo\data\444\im0.png")
    img_right = cv2.imread(r"D:\BaiduSyncdisk\work\Stereo\data\444\im1.png")
    img_left = cv2.resize(img_left, (img_left.shape[1]//2, img_left.shape[0]//2))
    img_right = cv2.resize(img_right, (img_right.shape[1]//2, img_right.shape[0]//2))
elif flag == 2:
    video_path = r"rectified_video.avi"
    # video_path = r"D:\BaiduSyncdisk\work\Stereo\data\stereo_video_20250325_170318.avi"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    img_left = frame[:frame.shape[0]//2, :, :]  # 上部分
    img_right = frame[frame.shape[0]//2:, :, :]  # 下部分
    img_left = cv2.resize(img_left, (640, 480))
    img_right = cv2.resize(img_right, (640, 480))
elif flag == 3:
    video1_path = r"D:\BaiduSyncdisk\work\Stereo\ONNX-Stereo-Project-zoo\calibration\rectified_left.avi"
    video2_path = r"D:\BaiduSyncdisk\work\Stereo\ONNX-Stereo-Project-zoo\calibration\rectified_right.avi"
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    ret1, img_left = cap1.read()
    ret2, img_right = cap2.read()
    img_left = cv2.resize(img_left, (640, 480))
    img_right = cv2.resize(img_right, (640, 480))
    # img_left = frame1[:frame1.shape[0]//2, :, :]
StereoRectify.rectify_video(img_left, img_right)
