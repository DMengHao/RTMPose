import cv2
import numpy as np
import sys
import os
import time

# 加入 rtmlib 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rtmlib import RTMPose, draw_skeleton

# 初始化姿态估计模型
pose_model = RTMPose(
    model='/home/dmh/hhkj/rtmlib-main/model/26_l.onnx',
    model_input_size=(192, 256),
    to_openpose=False,
    backend='onnxruntime',
    device='cpu'  # or 'cuda'
)

# 读取视频
video_path = '/home/dmh/hhkj/rtmlib-main/video/test.mp4'
cap = cv2.VideoCapture(video_path)

# 获取视频参数
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 保存输出视频（可选）
save_video = True
if save_video:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('pose_video_result.mp4', fourcc, fps, (width, height))

# 视频帧循环处理
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    bbox = np.array([[0, 0, width, height]])  # 假设整张图就是人体
    start = time.time()
    keypoints, scores = pose_model(frame, bboxes=bbox)
    elapsed = (time.time() - start) * 1000

    # 可视化
    frame_show = draw_skeleton(frame.copy(), keypoints, scores, kpt_thr=0.2)
    cv2.putText(frame_show, f"Inference: {elapsed:.1f} ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 显示
    cv2.imshow('Pose Estimation', cv2.resize(frame_show, (960, 540)))
    if save_video:
        out.write(frame_show)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
if save_video:
    out.release()
cv2.destroyAllWindows()
