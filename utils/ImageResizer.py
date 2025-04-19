# -*- coding: utf-8 -*-
# @Time    : 2025/4/19 22:57
# @Author  : sjh
# @Site    : 
# @File    : ImageResizer.py
# @Comment :
import cv2
import numpy as np


class ImageResizer:
    """图像缩放处理类，用于调整输入图像尺寸"""

    def __init__(self, target_width, target_height):
        # 目标宽度和高度
        self.target_width = target_width
        self.target_height = target_height
        # 记录原始图像尺寸,用于后续还原
        self.original_size = None
        # 记录填充的像素数,用于后续还原
        self.pad_top = 0
        self.pad_bottom = 0
        self.pad_left = 0
        self.pad_right = 0

    def resize_and_pad(self, img1, img2, divis_by=32.0):
        """调整图像尺寸并进行必要的填充

        Args:
            img1: 左图像,形状为[1,3,H,W]
            img2: 右图像,形状为[1,3,H,W]
            divis_by: 确保宽高能被此数整除,默认32

        Returns:
            处理后的左右图像元组
        """
        # 记录输入图像的原始尺寸,用于后续还原
        self.original_size = img1.shape[2:]
        _, _, h, w = img1.shape

        # 计算目标宽度,确保能被divis_by整除
        adjusted_target_width = round(self.target_width / divis_by) * divis_by

        # 如果调整后的宽度与原宽度相差小于divis_by,则保持原尺寸
        scale = 1.0 if abs(adjusted_target_width - w) < divis_by else adjusted_target_width / w

        # 计算新的宽高
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 只有在需要缩放时才进行resize操作
        if scale != 1.0:
            # 转换为HWC格式并调整大小
            img1_resized = cv2.resize(img1[0].transpose(1, 2, 0), (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            img2_resized = cv2.resize(img2[0].transpose(1, 2, 0), (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            # 不需要缩放时直接转换格式
            img1_resized = img1[0].transpose(1, 2, 0)
            img2_resized = img2[0].transpose(1, 2, 0)

        # 处理高度方向的填充或裁剪
        if new_h < self.target_height:
            # 填充
            self.pad_top = (self.target_height - new_h) // 2
            self.pad_bottom = self.target_height - new_h - self.pad_top
            self.pad_left = (self.target_width - new_w) // 2
            self.pad_right = self.target_width - new_w - self.pad_left

            # 使用边缘填充模式
            img1_processed = np.pad(img1_resized, (
            (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right), (0, 0)),
                                    mode='edge')
            img2_processed = np.pad(img2_resized, (
            (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right), (0, 0)),
                                    mode='edge')
        else:
            # 裁剪
            self.pad_top = (new_h - self.target_height) // 2
            img1_processed = img1_resized[self.pad_top:self.pad_top + self.target_height, :, :]
            img2_processed = img2_resized[self.pad_top:self.pad_top + self.target_height, :, :]

        # 转换回NCHW格式并确保类型为float32
        return (img1_processed.transpose(2, 0, 1)[None].astype(np.float32),
                img2_processed.transpose(2, 0, 1)[None].astype(np.float32))

    def reverse_process(self, flow, divis_by=32.0):
        """还原处理后的视差图到原始尺寸

        Args:
            flow: 模型输出的视差图
            divis_by: 与resize_and_pad中使用的相同值

        Returns:
            还原到原始尺寸的视差图
        """
        h, w = self.original_size
        # 使用与resize_and_pad相同的方式计算缩放比例
        adjusted_target_width = round(self.target_width / divis_by) * divis_by
        scale = adjusted_target_width / w
        new_h = int(h * scale)

        # 处理填充还原
        if new_h < self.target_height:
            # 如果之前是填充,现在移除填充
            flow = flow[self.pad_top:-self.pad_bottom] if self.pad_bottom > 0 else flow[self.pad_top:]
        else:
            # 如果之前是裁剪,现在需要补充回原始高度
            flow = np.pad(flow, ((self.pad_top, new_h - self.target_height - self.pad_top), (0, 0)), mode='constant')

        # 还原到原始尺寸并调整视差值
        flow_resized = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        # 根据缩放比例调整视差值,因为视差值与图像宽度成正比
        flow_resized = flow_resized * (w / self.target_width)

        return flow_resized
