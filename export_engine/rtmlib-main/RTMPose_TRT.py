import cv2
import numpy as np
import sys
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
from rtmlib import RTMPose, draw_skeleton





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
                    output_size: [int, int],
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
    center = center[0]
    scale = scale[0]
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

def top_down_affine_tensor(input_size, bbox_scale, bbox_center, img):
    w, h = input_size
    warp_size = (int(w), int(h))
    aspect_ratio = w / h
    b_w, b_h = torch.split(torch.from_numpy(bbox_scale),[1,1],dim=1)




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


def get_simcc_maximum(simcc_x: np.ndarray,
                      simcc_y: np.ndarray):
    """Get maximum response location and value from simcc representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    # get maximum value locations
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    # get maximum value across x and y axis
    # mask = max_val_x > max_val_y
    # max_val_x[mask] = max_val_y[mask]
    vals = 0.5 * (max_val_x + max_val_y)
    locs[vals <= 0.] = -1

    # reshape
    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals

def bbox_xyxy2cs(bbox: np.ndarray,
                 padding: float = 1.):
    # convert single bbox from (4, ) to (1, 4)
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    # get bbox center and scale
    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale
    
class rtmlib_trt:
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

        # get center and scale
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)

        # do affine transformation
        resized_img, scale = top_down_affine(self.input_size, scale,
                                                center, img)
        mean: tuple = (123.675, 116.28, 103.53),
        std: tuple = (58.395, 57.12, 57.375),
        # normalize image
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
        """Postprocess for RTMPose model output.

        Args:
            outputs (np.ndarray): Output of RTMPose model.
            model_input_size (tuple): RTMPose model Input image size.
            center (tuple): Center of bbox in shape (x, y).
            scale (tuple): Scale of bbox in shape (w, h).
            simcc_split_ratio (float): Split ratio of simcc.

        Returns:
            tuple:
            - keypoints (np.ndarray): Rescaled keypoints.
            - scores (np.ndarray): Model predict scores.
        """
        # decode simcc
        simcc_x, simcc_y = outputs
        locs, scores = get_simcc_maximum(simcc_x, simcc_y)
        keypoints = locs / simcc_split_ratio

        # rescale keypoints
        keypoints = keypoints / self.input_size * scale
        keypoints = keypoints + center - scale / 2

        return keypoints, scores


    def infer(self, tensor_data):
        try:
            self.ctx.push()
            self.binding_addrs['input'] = int(tensor_data.data_ptr())
            temp_time = time.time()
            self.context.execute_v2(list(self.binding_addrs.values()))
            self.stream.synchronize()  # 确保所有计算已完成，阻塞调用
            preds = [self.bindings['simcc_x'].data.detach().cpu().numpy(), self.bindings['simcc_y'].data.detach().cpu().numpy()]
            return preds
        finally:
            self.ctx.pop()


    def __del__(self):
        self.ctx.pop()
        del self.context
        del self.model
        del self.runtime


# 初始化模型
pose_model = RTMPose(
    onnx_model='/home/hhkj/dmh/RZG/rtmlib-main/model/17_l.onnx',  # ⚠️ 注意参数应为 model，不是 onnx_model
    model_input_size=(192, 256),
    to_openpose=False,
    backend='onnxruntime',
    device='cpu'  # or 'cuda'
)


img = cv2.imread('/home/hhkj/dmh/RZG/rtmlib-main/images/demo.jpg')
img_copy = img.copy()
h, w, _ = img.shape

bbox = np.array([[0, 0, w, h]])  # 假设整张图为人体框

RTMPose_TRT = rtmlib_trt(weights='/home/hhkj/dmh/RZG/rtmlib-main/model/17_l.engine', dev='cuda:0')
for i in range(50):
    all_time = time.time()
    preprocess_time = time.time()
    img, center, scale = RTMPose_TRT.preprocess(img, bbox)
    print(f"前处理推理耗时: {(time.time()-preprocess_time)*1000:.2f} ms")
    engine_time = time.time()
    outputs = RTMPose_TRT.infer(torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)[None, :, :, :]).to('cuda'))
    print(f"TensorRT推理耗时: {(time.time()-engine_time)*1000:.2f} ms")
    postprocess_time = time.time()
    kpts, score = RTMPose_TRT.postprocess(outputs, center[0], scale[0])
    print(f"后处理推理耗时: {(time.time()-postprocess_time)*1000:.2f} ms")
    print(f"总耗时: {(time.time()-all_time)*1000:.2f} ms")
    keypoints = np.concatenate([kpts], axis=0)
    scores = np.concatenate([score], axis=0)

    # keypoints, scores = pose_model(img_copy, bboxes=bbox)

    img_show = draw_skeleton(img_copy, keypoints, scores, kpt_thr=0.2)

    cv2.imwrite('./save_path.jpg', img_show)
    print('测试完成！')