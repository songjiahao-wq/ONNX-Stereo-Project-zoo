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
video_low_filename = os.path.join('./', "depth_video_low.avi")
video_writer_high = cv2.VideoWriter(video_high_filename, fourcc, fps, frame_size, isColor=False)
video_writer_low = cv2.VideoWriter(video_low_filename, fourcc, fps, frame_size, isColor=False)
def delta_encode(data):
    """差分编码（第一列保留原值，后续列存储差值）"""
    diff = np.zeros_like(data)
    diff[:, 0] = data[:, 0]  # 第一列保持原值
    diff[:, 1:] = data[:, 1:] - data[:, :-1]  # 后续列存储差值
    return diff
try:
    while cap_high.isOpened() and cap_low.isOpened():
        ret_high, frame_high = cap_high.read(cv2.IMREAD_UNCHANGED)
        ret_low, frame_low = cap_low.read()

        if not ret_high or not ret_low:
            break

        # 将高 8 位和低 8 位组合回 uint16
        frame_uint16 = (frame_high.astype(np.uint16) << 8) | frame_low.astype(np.uint16)
        cv2.imshow('Reconstructed Depth Image', frame_uint16.astype(np.float32) / 65535.0)

        # 将高位和低位图像保存为单独的视频文件
        high_bits = (frame_uint16 >> 8).astype(np.uint8)  # 高 8 位
        low_bits = (frame_uint16 & 0xFF).astype(np.uint8)  # 低 8 位
        low_bits_shifted = (low_bits >> 2).astype(np.uint8) # 将低八位右移一位
        high_bits = high_bits[:, :, 0]  # 取第一个通道
        low_bits_shifted = low_bits_shifted[:, :, 0]  # 取第一个通道
        low_bits_shifted = low_bits_shifted / 2
        # low_bits_shifted = delta_encode(low_bits_shifted)
        print(high_bits.shape)
        print("High bits range:", np.min(high_bits), np.max(high_bits))  # 应为 0-255
        print("Low bits range:", np.min(low_bits_shifted), np.max(low_bits_shifted))  # 应为 0-255
        video_writer_high.write(high_bits)
        video_writer_low.write(low_bits_shifted)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print("Error:", e)
finally:
    video_writer_high.release()
    video_writer_low.release()