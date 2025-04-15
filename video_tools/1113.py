import numpy as np
import cv2

# 生成模拟深度图 (480x640, 16位无符号整数)
depth_map = np.random.randint(0, 65536, size=(480, 640), dtype=np.uint16)

# 添加一些模拟的物体（例如：一个矩形区域代表近距离物体）
depth_map[100:300, 200:400] = 3000  # 3000mm = 3米
depth_map[350:450, 100:300] = 8000   # 8000mm = 8米

# 拆分高 8 位和低 8 位
high_byte = (depth_map >> 8).astype(np.uint8)  # 取高 8 位
low_byte = (depth_map & 0xFF).astype(np.uint8) # 取低 8 位

# 显示高 8 位和低 8 位
cv2.imshow("High Byte (MSB)", high_byte)
cv2.imshow("Low Byte (LSB)", low_byte)
cv2.waitKey(0)