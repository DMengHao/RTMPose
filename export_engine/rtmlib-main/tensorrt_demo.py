# -*- coding:utf-8 -*-
import os
import time
import logging
import cv2
import numpy as np
import torch
import tensorrt as trt
from pycuda import driver
import pycuda.driver as cuda0
from collections import OrderedDict, namedtuple
from tqdm import tqdm
import pycuda.autoinit
import yaml
from pathlib import Path
import sys
from post_processings import convert_coco_to_openpose, get_simcc_maximum
from pre_processings import bbox_xyxy2cs, top_down_affine
from rtmlib import draw_skeleton

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH 为了导入下方的models和utils
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# 创建日志记录器并设置级别
logger = logging.getLogger('crossingdemo')
logger.setLevel(logging.INFO)
# 定义日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 创建一个控制台输出的处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
# 添加处理器
logger.addHandler(console_handler)


def top_down_affine(input_size: dict, bbox_scale: dict, bbox_center: dict,
                    img: np.ndarray):
    """Get the bbox image as the model input by affine transform.

    Args:
        input_size (dict): The input size of the model.
        bbox_scale (dict): The bbox scale of the img.
        bbox_center (dict): The bbox center of the img.
        img (np.ndarray): The original image.

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: img after affine transform.
        - np.ndarray[float32]: bbox scale after affine transform.
    """
    w, h = input_size
    warp_size = (int(w), int(h))

    # reshape bbox to fixed aspect ratio
    aspect_ratio = w / h
    bbox_scale = np.array(bbox_scale)
    b_w, b_h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(b_w > b_h * aspect_ratio,
                          np.hstack([b_w, b_w / aspect_ratio]),
                          np.hstack([b_h * aspect_ratio, b_h]))

    # get the affine matrix
    center = bbox_center
    scale = bbox_scale
    rot = 0
    warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

    # do affine transform
    img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)

    return img, bbox_scale


def bbox_xyxy2cs(bbox: np.ndarray,
                 padding: float = 1.):
    """Transform the bbox format from (x,y,w,h) into (center, scale)

    Args:
        bbox (ndarray): Bounding box(es) in shape (4,) or (n, 4), formatted
            as (left, top, right, bottom)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: Center (x, y) of the bbox in shape (2,) or
            (n, 2)
        - np.ndarray[float32]: Scale (w, h) of the bbox in shape (2,) or
            (n, 2)
    """
    center = [[(bbox[2]-bbox[0])/2, (bbox[3]-bbox[1])/2]]
    scale = [(bbox[2]-bbox[0])* padding, (bbox[3]-bbox[1])* padding] 
    # # convert single bbox from (4, ) to (1, 4)
    # dim = bbox.ndim
    # if dim == 1:
    #     bbox = bbox[None, :]

    # # get bbox center and scale
    # x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    # center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    # scale = np.hstack([x2 - x1, y2 - y1]) * padding

    # if dim == 1:
    #     center = center[0]
    #     scale = scale[0]

    return center, scale


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a point by an angle.

    Args:
        pt (np.ndarray): 2D point coordinates (x, y) in shape (2, )
        angle_rad (float): rotation angle in radian

    Returns:
        np.ndarray: Rotated point in shape (2, )
    """
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt

def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): The 1st point (x,y) in shape (2, )
        b (np.ndarray): The 2nd point (x,y) in shape (2, )

    Returns:
        np.ndarray: The 3rd point.
    """
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c

def get_warp_matrix(center: np.ndarray,
                    scale: np.ndarray,
                    rot: float,
                    output_size: [],
                    shift: [float, float] = (0., 0.),
                    inv: bool = False) :
    """Calculate the affine transformation matrix that can warp the bbox area
    in the input image to the output size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: A 2x3 transformation matrix
    """

    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    # compute transformation matrix
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0., src_w * -0.5]), rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    # get four corners of the src rectangle in the original image
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    # get four corners of the dst rectangle in the input image
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]

    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return warp_mat


class rtmlib_trt:
    def __init__(self, weights='/home/hhkj/dmh/RZG/rtmlib-main/model/17_l.engine', dev='cuda:0', half=True, input_size=(192, 256), to_openpose=False):

        self.input_size = input_size
        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])
        self.to_openpose = to_openpose

        self.device = int(dev.split(':')[-1])
        self.ctx = cuda0.Device(self.device).make_context()
        self.stream = driver.Stream()
        self.names = []
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

    def preprocess(self, img, bbox):
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)
        resized_img, scale = top_down_affine(self.input_size, scale, center, img)
        resized_img = (resized_img - self.mean) / self.std
        resized_img = resized_img.transpose(2, 0, 1)[np.newaxis, :]
        return torch.from_numpy(resized_img.astype(np.float32)).to('cuda:0'), center, scale

    def postprocess(self, outputs, center, scale):
        scale = np.array(scale)
        center = np.array(center)
        simcc_x, simcc_y = outputs
        locs, scores = get_simcc_maximum(simcc_x, simcc_y)
        keypoints = locs / 2.0
        keypoints = keypoints / self.input_size * scale
        # keypoints = keypoints + center - scale / 2
        return keypoints, scores

    def infer(self, tensor_data):
        try:
            self.ctx.push()
            infer_start_time = time.time()
            self.binding_addrs['input'] = int(tensor_data.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            self.stream.synchronize()  # 确保所有计算已完成，阻塞调用
            preds = [self.bindings['simcc_x'].data.detach().cpu().numpy(), self.bindings['simcc_y'].data.detach().cpu().numpy()]
            logger.info(f'推理用时：{(time.time() - infer_start_time) * 1000:.4f}ms')
            return preds
        finally:
            self.ctx.pop()

    def __call__(self, img, bbox):
        input_tensor, center, scale = self.preprocess(img, bbox)
        center = [475, 320.5]
        scale = [1187.5, 1583]
        outputs = self.infer(input_tensor)
        keypoints, scores = self.postprocess(outputs, center, scale)
        return keypoints, scores

    def __del__(self):
        self.ctx.pop()
        del self.context
        del self.model
        del self.runtime


if __name__ == '__main__':
    # 加载模型
    RTMPose_TRT = rtmlib_trt('/home/hhkj/dmh/RZG/rtmlib-main/model/17_l.engine')

    # 包装成 RTMPose 结构
    # pose_model = RTMPose_TRT(rtmlib_trt_api)

    # 读取图片
    image_path = '/home/hhkj/dmh/RZG/crossingdemo/box_image/73_0.jpg'
    img = cv2.imread(image_path)

    # 假设整张图是一个框（或替换成你检测出来的 bbox）
    bbox = [0, 0, img.shape[1], img.shape[0]]  # xyxy

    # 推理关键点
    keypoints, scores = RTMPose_TRT(img, bbox)

    img_show = draw_skeleton(img, keypoints, scores, kpt_thr=0.2)

    # save_path = os.path.join(save_folder, filename)
    cv2.imwrite('/home/hhkj/dmh/RZG/rtmlib-main/results.jpg', img_show)