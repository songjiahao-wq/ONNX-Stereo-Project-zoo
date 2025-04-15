# -*- coding: utf-8 -*-
# @Time    : 2025/4/7 21:56
# @Author  : sjh
# @Site    :
# @File    : main.py
# @Comment :
from save import *
input_high_path  = r'D:\BaiduSyncdisk\work\data\20250401152349\action_video/depth_action_2_high.avi'
input_low_path  = r'D:\BaiduSyncdisk\work\data\20250401152349\action_video/depth_action_2_low.avi'
cap_high = cv2.VideoCapture(input_high_path)
cap_low = cv2.VideoCapture(input_low_path)

fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # 使用 FFV1 编解码器
fps = 30  # 每秒帧数
frame_size = (512, 424)  # 深度图像的尺寸 (宽度, 高度)
# frame_size = (424, 512)  # 深度图像的尺寸 (宽度, 高度)
video_high_filename = os.path.join('./', "depth_video_high.avi")
video_writer_high = cv2.VideoWriter(video_high_filename, fourcc, fps, frame_size, isColor=True)

try:
    while cap_high.isOpened() and cap_low.isOpened():
        ret_high, frame_high = cap_high.read(cv2.IMREAD_UNCHANGED)
        ret_low, frame_low = cap_low.read()

        if not ret_high or not ret_low:
            break

        # 将高 8 位和低 8 位组合回 uint16
        frame_uint16 = (frame_high.astype(np.uint16) << 8) | frame_low.astype(np.uint16)
        cv2.imshow('Reconstructed Depth Image', frame_uint16.astype(np.float32) / 65535.0)
        frame_uint16 = frame_uint16[:, :, 0]  # 取第一个通道
        # 将16位数据分解到3个通道
        channel_r = (frame_uint16 >> 8).astype(np.uint8)  # 高8位
        channel_g = ((frame_uint16 & 0xF0) >> 4).astype(np.uint8)  # 中4位
        channel_b = (frame_uint16 & 0x0F).astype(np.uint8)  # 低4位
        print(np.max(channel_r), np.min(channel_r))
        print(np.max(channel_g), np.min(channel_g))
        print(np.max(channel_b), np.min(channel_b))
        print('aaaaaaa')
        # 合并为3通道图像
        multi_channel = cv2.merge([channel_r, channel_g, channel_b])
        print(multi_channel.shape)
        video_writer_high.write(multi_channel)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print("Error:", e)
finally:
    video_writer_high.release()
