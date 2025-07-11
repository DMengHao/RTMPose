import cv2
import numpy as np
import sys
import os
import time

# 加载姿态估计模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rtmlib import RTMPose, draw_skeleton

# 初始化模型
pose_model = RTMPose(
    onnx_model='/home/hhkj/dmh/RZG/rtmlib-main/model/17_l.onnx',  # ⚠️ 注意参数应为 model，不是 onnx_model
    model_input_size=(192, 256),
    to_openpose=False,
    backend='onnxruntime',
    device='cpu'  # or 'cuda'
)

# 设置输入输出路径
img_folder = '/home/hhkj/dmh/RZG/rtmlib-main/images/'         # 输入图片文件夹
save_folder = '/home/hhkj/dmh/RZG/rtmlib-main/results/'       # 输出结果保存路径
os.makedirs(save_folder, exist_ok=True)

# 支持的图片格式
img_exts = ('.jpg', '.jpeg', '.png', '.bmp')

for i in range(50):
    # 遍历文件夹下所有图片
    for filename in sorted(os.listdir(img_folder)):
        if not filename.lower().endswith(img_exts):
            continue  # 忽略非图片文件

        img_path = os.path.join(img_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"跳过无效图像: {filename}")
            continue

        h, w, _ = img.shape
        bbox = np.array([[0, 0, w, h]])  # 假设整张图为人体框

        start_time = time.time()
        keypoints, scores = pose_model(img, bboxes=bbox)
        # np.savetxt(os.path.join(save_folder,os.path.splitext(filename)[0]+'.txt'), keypoints[0],fmt='%.6f', delimiter=' ')
        elapsed = (time.time() - start_time) * 1000

        print(f"{filename} 推理耗时: {elapsed:.2f} ms")

        img_show = draw_skeleton(img, keypoints, scores, kpt_thr=0.2)

        save_path = os.path.join(save_folder, filename)
        cv2.imwrite(save_path, img_show)
