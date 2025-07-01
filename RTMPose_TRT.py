import cv2
import numpy as np
import sys
from tqdm import tqdm
import os
import time
import pycuda.driver as cuda0
import pycuda.autoinit
from pycuda import driver
import tensorrt as trt
import torch
from collections import OrderedDict, namedtuple
# 加载姿态估计模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from rtmlib import draw_skeleton
from utils.utils import bbox_xyxy2cs, top_down_affine, _rotate_point, _get_3rd_point, get_warp_matrix, top_down_affine_tensor, get_simcc_maximum, prepare_coco_keypoints, draw_coco_keypoints


class RTMPose_TRT:
    def __init__(self, weights='/home/hhkj/dmh/RZG/rtmlib-main/model/17_l.engine', dev='cuda:0', half=True, input_size=(192, 256), to_openpose=False):
        self.input_size = input_size
        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])
        self.to_openpose = to_openpose
        self.device = int(dev.split(':')[-1])
        self.ctx = cuda0.Device(self.device).make_context()
        self.stream = driver.Stream()
        logger = trt.Logger(trt.Logger.INFO)
        with open(weights, 'rb') as f:
            self.runtime = trt.Runtime(logger)
            self.model = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.model.create_execution_context()
        self.bindings = OrderedDict()
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        for index in range(self.model.num_bindings):
            if trt.__version__ >= '8.6.1':
                name = self.model.get_tensor_name(index)
                dtype = trt.nptype(self.model.get_tensor_dtype(name))
                shape = tuple(self.model.get_tensor_shape(name))
            else:
                name = self.model.get_binding_name(index)
                dtype = trt.nptype(self.model.get_binding_dtype(index))
                shape = tuple(self.model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            del data
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())


    def preprocess(self, img: np.ndarray, bbox: list):
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)
        resized_img, scale = top_down_affine(self.input_size, scale, center, img)
        mean: tuple = (123.675, 116.28, 103.53),
        std: tuple = (58.395, 57.12, 57.375),
        # 归一化输入
        if mean is not None:
            mean = np.array(mean)
            std = np.array(std)
            resized_img = (resized_img - mean) / std
        return resized_img, center, scale


    def postprocess(
            self,
            outputs: [],
            center: [int, int],
            scale: [int, int],
            simcc_split_ratio: float = 2.0):
        # decode simcc
        simcc_x, simcc_y = outputs
        locs, scores = get_simcc_maximum(simcc_x, simcc_y)
        keypoints = locs / simcc_split_ratio
        # rescale keypoints
        keypoints = keypoints / self.input_size * scale
        keypoints = keypoints + center - scale / 2
        return keypoints, scores


    def infer(self, img, bbox):
        try:
            self.ctx.push()
            pre_time = time.time()
            img, center, scale = self.preprocess(img, bbox)
            print(f'RTMPoseTRT前处理耗时：{(time.time()-pre_time)*1000:.2f}ms')
            infer_time = time.time()
            self.binding_addrs['input'] = int(torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)[None, :, :, :]).to('cuda').data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            self.stream.synchronize()  # 确保所有计算已完成，阻塞调用
            preds = [self.bindings['simcc_x'].data.detach().cpu().numpy(), self.bindings['simcc_y'].data.detach().cpu().numpy()]
            print(f'RTMPoseTRT推理耗时：{(time.time()-infer_time)*1000:.2f}ms')
            post_time = time.time()
            kpts, score = self.postprocess(preds, center[0], scale[0])
            keypoints = np.concatenate([kpts], axis=0)
            scores = np.concatenate([score], axis=0)
            print(f'RTMPose推理耗时：{(time.time()-post_time)*1000:.2f}ms')
            return keypoints, scores
        finally:
            self.ctx.pop()


    def __del__(self):
        self.ctx.pop()
        del self.context
        del self.model
        del self.runtime
        
if __name__ == '__main__':
    RTMPose_TRT_api = RTMPose_TRT(weights='/home/hhkj/dmh/RZG/rtmlib-main/model/17_l.engine', dev='cuda:0')
    IMAGE_PATH = "/home/hhkj/dmh/RZG/crossingdemo/box_image/"  # 待检测图片路径
    OUTPUT_PATH = "/home/hhkj/dmh/RZG/crossingdemo/results/"  # 输出图片路径
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    image_path = os.listdir(IMAGE_PATH)
    print('开始测试：')
    for i,img_path in tqdm(enumerate(image_path), total=len(image_path)):
        image = cv2.imdecode(np.fromfile(os.path.join(IMAGE_PATH,img_path), dtype=np.uint8), -1)
        # image = cv2.resize(image, (image.shape[0]*2, image.shape[1]*2), interpolation=cv2.INTER_CUBIC)
        img_copy = image.copy()
        h, w, _ = img_copy.shape
        bbox = np.array([[0, 0, w, h]])  # 假设整张图为人体框
        keypoints, scores = RTMPose_TRT_api.infer(img_copy, bbox)
        # 绘制关键点
        result_img = draw_coco_keypoints(img_copy, keypoints, scores, confidence_threshold=0.2)
        cv2.imwrite(os.path.join(OUTPUT_PATH,img_path), result_img)

    print('测试完毕！')

