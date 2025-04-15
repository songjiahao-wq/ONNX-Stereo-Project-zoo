import cv2
import numpy as np

# 左右相机的内参和畸变系数
K_left = np.array([[745.36569178, 0., 540.56511955],
                   [0., 746.73646007, 386.53044259],
                   [0., 0., 1.]])  # 左相机内参

K_right = np.array([[743.30812943, 0., 541.17968337],
                    [0., 744.44363451, 395.80555331],
                    [0., 0., 1.]])  # 右相机内参

dist_left = np.array([1.37253265e-01, -4.59574066e-01, -1.81794070e-04, 1.86765804e-03, 4.20309007e-01])  # 左相机畸变系数
dist_right = np.array([1.37253265e-01, -4.59574066e-01, -1.81794070e-04, 1.86765804e-03, 4.20309007e-01])  # 右相机畸变系数

# 旋转矩阵和位移向量（从双目标定结果中获得）
R = np.array([[ 0.99991539,  0.00930727, -0.00908758],
              [-0.0093278 ,  0.99995403, -0.00221897],
              [ 0.00906651,  0.00230354,  0.99995625]])

T = np.array([[-0.10639568], [-0.00107478], [-0.01026687]])





# 读取视频
cap_left = cv2.VideoCapture(r"D:\BaiduSyncdisk\work\Stereo\stereo_test_optimized\data\ai\0305\20250305133700\step2\a0.avi")
cap_right = cv2.VideoCapture(r"D:\BaiduSyncdisk\work\Stereo\stereo_test_optimized\data\ai\0305\20250305133700\step2\a1.avi")

# 获取视频的帧率
fps = cap_left.get(cv2.CAP_PROP_FPS)

# 获取视频尺寸（假设两张图像尺寸相同）
width = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建视频写入器
fourcc = cv2.VideoWriter_fourcc(*'FFV1')
out_left = cv2.VideoWriter('rectified_combined.avi', fourcc, fps, (width, height * 2))

# 计算立体校正变换
# 这里我们使用 cv2.stereoRectify 来计算左右图像的变换矩阵和映射矩阵
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K_left, dist_left, K_right, dist_right, (width, height), R, T, alpha=0)
cx = -Q[0, 3]
cy = -Q[1, 3]
fx = Q[2, 3]
fy = Q[2, 3]
baseline = abs(1.0 / Q[3, 2])

print(f"fx={fx}, fy={fy}, cx={cx}, cy={cy}, baseline={baseline}")
# 获取视差图像的重新映射
map1x, map1y = cv2.initUndistortRectifyMap(K_left, dist_left, R1, P1, (width, height), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K_right, dist_right, R2, P2, (width, height), cv2.CV_32FC1)

while cap_left.isOpened() and cap_right.isOpened():
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    
    if not ret_left or not ret_right:
        break
    # 去畸变处理
    # new_mtx_left, roi_left = cv2.getOptimalNewCameraMatrix(K_left, dist_left, (width, height), 1, (width, height))
    # new_mtx_right, roi_right = cv2.getOptimalNewCameraMatrix(K_right, dist_right, (width, height), 1, (width, height))
    # frame_left = cv2.undistort(frame_left, K_left, dist_left, None, new_mtx_left)
    # frame_right = cv2.undistort(frame_right, K_right, dist_right, None, new_mtx_right)
    
    # 将每个图像进行重新映射，进行立体校正
    rectified_left = cv2.remap(frame_left, map1x, map1y, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(frame_right, map2x, map2y, cv2.INTER_LINEAR)
    combined_img = np.vstack((rectified_left, rectified_right))
    
    # 将校正后的图像保存到输出视频文件
    out_left.write(combined_img)
    
    # combined_img = cv2.resize(combined_img, (, 768))
    # 显示校正后的图像
    cv2.imshow('Rectified Left', combined_img)
    # cv2.imwrite('rectified_left.png', rectified_left)
    # cv2.imwrite('rectified_right.png', rectified_right)
    # break
    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获和写入对象
cap_left.release()
cap_right.release()
out_left.release()

cv2.destroyAllWindows()
