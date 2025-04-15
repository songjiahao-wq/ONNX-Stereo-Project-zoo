a0 = r"D:\BaiduSyncdisk\work\Stereo\data\AI\20250305133700\step1\a0.avi"
a1 = r"D:\BaiduSyncdisk\work\Stereo\data\AI\20250305133700\step1\a1.avi"

import cv2
import os
import shutil
def extract_frames(video_path, output_dir):
    # 创建输出目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)  
    
    frame_count = 0
    get_frame_count = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        if frame_count % 15 == 0:
            # 保存帧为图片
            frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_path, frame)
            get_frame_count += 1
        frame_count += 1
        
    cap.release()
    
    print(f"提取完成，共提取了{get_frame_count}帧")
    
if __name__ == "__main__":
    # 创建输出目录
    output_dir = r"data/left"
    extract_frames(a0, output_dir)
    
    output_dir = r"data/right"
    extract_frames(a1, output_dir)

