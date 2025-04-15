# -*- coding: utf-8 -*-
# @Time    : 2025/4/7 21:56
# @Author  : sjh
# @Site    :
# @File    : main.py
# @Comment :
from save import *
input_low_path  = r'D:\BaiduSyncdisk\work\data\20250401152349\action_video/depth_action_2_low.avi'
cap_low = cv2.VideoCapture(input_low_path)

fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # 使用 FFV1 编解码器
fps = 30  # 每秒帧数
frame_size = (512, 424)  # 深度图像的尺寸 (宽度, 高度)
# frame_size = (424, 512)  # 深度图像的尺寸 (宽度, 高度)
video_low_filename = os.path.join('./', "depth_video_low.avi")
video_writer_low = cv2.VideoWriter(video_low_filename, fourcc, fps, frame_size, isColor=False)

try:
    while cap_low.isOpened():
        ret_low, frame_low = cap_low.read()

        if not ret_low:
            break

        # 将高 8 位和低 8 位组合回 uint16
        frame_low = frame_low.astype(np.uint8)
        cv2.imshow('Reconstructed Depth Image', frame_low.astype(np.float32) / 65535.0)

        # 将高位和低位图像保存为单独的视频文件
        low_bits_shifted = (frame_low >> 4).astype(np.uint8) # 将低八位右移一位
        low_bits_shifted = low_bits_shifted[:, :, 0]  # 取第一个通道
        print(low_bits_shifted.shape)
        print("Low bits range:", np.min(low_bits_shifted), np.max(low_bits_shifted))  # 应为 0-255
        video_writer_low.write(low_bits_shifted)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print("Error:", e)
finally:
    video_writer_low.release()