# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import torch
import torchvision
from torch.nn import functional as F
from torchvision.transforms import Resize
import time
from pathlib import Path
import sys
import os
from torchvision.ops import box_iou

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def save_pic(image, save_path):
    cv2.imencode('.jpg', image)[1].tofile(save_path)


def draw_pic(img_read, text, x1, y1, x2, y2):
    cv2.rectangle(img_read, (x1, y1), (x2, y2), (0, 0, 220), 2)
    cv2.putText(img_read, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 220), 2)


def draw(img_tensor, detections, img_names, image_name):
    save_path = ROOT /'results/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i, img in enumerate(img_tensor):
        image = img.cpu().numpy().astype(np.uint8)
        if os.path.splitext(img_names[i])[1] in ['.jpg', '.JPG', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.svg', '.pfg']:
            for item in detections[i]:
                text = item['class']
                box = item['box']
                conf = item['conf']
                label = f'{text} ({conf:.2f})'
                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
                draw_pic(image, label, x1, y1, x2, y2)  # 使用方法打包汇框和标签，便于维护
            pic_save_path = f'{save_path}/{image_name+os.path.basename(img_names[i])}'
            save_pic(image, pic_save_path)
            print(f'保存推理结果图路径：{pic_save_path}')


def IoU(box1, box2) -> float:
    weight = max(min(box1[2], box2[2]) - max(box1[0], box2[0]), 0)
    height = max(min(box1[3], box2[3]) - max(box1[1], box2[1]), 0)
    s_inter = weight * height
    s_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    s_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    s_union = s_box1 + s_box2 - s_inter
    return s_inter / s_union

def nms_task1(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    min_wh, max_wh = 2, 7680
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    multi_label &= nc > 1
    merge = False

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0
        x = x[xc[xi]]
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        # 0.2375ms vs 0.1187ms
        # idxs = torch.arange(boxes.shape[0], device=boxes.device)
        # i = torchvision.ops.batched_nms(boxes=boxes, scores=scores, idxs=idxs, iou_threshold=iou_thres)
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and (1 < n < 3E3):
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break
    return output

def nms(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    min_wh, max_wh = 2, 7680
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    multi_label &= nc > 1
    merge = False

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0
        x = x[xc[xi]]
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        # 0.2375ms vs 0.1187ms
        # idxs = torch.arange(boxes.shape[0], device=boxes.device)
        # i = torchvision.ops.batched_nms(boxes=boxes, scores=scores, idxs=idxs, iou_threshold=iou_thres)
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and (1 < n < 3E3):
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break
    return output


def clip_coords(boxes, shape):
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[1])
        boxes[:, 1].clamp_(0, shape[0])
        boxes[:, 2].clamp_(0, shape[1])
        boxes[:, 3].clamp_(0, shape[0])
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]) # 计算输入图像相较于原始图像的缩放比例
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def letterbox_xu(im, new_shape=(640, 640), auto=True, scaleFill=False, scaleup=True, stride=32): # im->(B,3,1280,1280)
    shape = im.shape[2:]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) #四舍五入，取整
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        torch_resize = Resize([new_unpad[1], new_unpad[0]])
        im = torch_resize(im)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    p2d = (left, right, top, bottom)
    out = F.pad(im, p2d, 'constant', 114.0 / 255.0)
    out = out.contiguous()
    return out, ratio, (dw, dh)

def draw(img, detections, save_path, color_map):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    image = img.copy()
    if os.path.splitext(save_path)[1] in ['.jpg', '.JPG', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.svg', '.pfg']:
        for item in detections:
            # text = item['class']+'_x_'+str(item['box'][2]-item['box'][0])+'_y_'+str(item['box'][3]-item['box'][1])
            text = item['class']
            box = item['box']
            conf = item['conf']
            label = f'{text} ({conf:.2f})'
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            color = color_map.get(item['class'], [np.random.randint(0, 255) for _ in range(2)])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            # 道口一
            if False:
                pts = np.array( [
                            [ 591, 443 ],
                            [ 815, 335 ],
                            [ 2543, 813 ],
                            [ 2553, 1275 ],
                            [ 1727, 1270 ],
                            [ 1731, 1430 ],
                            [ 1274, 1436 ],
                            [ 372, 537 ]
                        ])
                pts = pts.reshape((-1,1,2))
                cv2.polylines(image,[pts], isClosed=True, color=(0,255,0),thickness=2)
            # 道口二
            if False:
                pts1 = np.array( [
                                [ 17, 831 ],
                                [ 898, 383 ],
                                [ 1419, 480 ],
                                [ 1120, 1067 ]
                            ])
                pts1 = pts1.reshape((-1,1,2))
                cv2.polylines(image,[pts1], isClosed=True, color=(0,255,0),thickness=2)
                pts2 = np.array([
                        [ 17, 831 ],
                        [ 591, 594 ],
                        [ 1307, 698 ],
                        [ 1120, 1067 ]
                    ])
                pts2 = pts2.reshape((-1,1,2))
                cv2.polylines(image,[pts2], isClosed=True, color=(0,255,0),thickness=2)
            # 道口三
            if False:
                pts1 = np.array( [
                                [ 12, 631 ],
                                [ 953, 485 ],
                                [ 1535, 496 ],
                                [ 1088, 1010 ],
                                [ 9, 1011 ]
                            ])
                pts1 = pts1.reshape((-1,1,2))
                cv2.polylines(image,[pts1], isClosed=True, color=(0,255,0),thickness=2)
                pts2 = np.array([
                        [ 12, 631 ],
                        [ 501, 555 ],
                        [ 1399, 662 ],
                        [ 1040, 1019 ],
                        [ 17, 1019 ]
                    ])
                pts2 = pts2.reshape((-1,1,2))
                cv2.polylines(image,[pts2], isClosed=True, color=(255,0,0),thickness=2)
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,color, 2)
        save_pic(image, save_path)

def save_box_image(img, detections, save_path):
    for i, detection in enumerate(detections):
        boxs = detection['box']
        crop = img[boxs[1]-4:boxs[3]+4,boxs[0]-4:boxs[2]+4,:]
        save_path = os.path.dirname(ROOT) + f'/box_image/{save_path}_{i}.jpg'
        if not os.path.exists(os.path.dirname(ROOT) + '/box_image'):
            os.makedirs(os.path.dirname(ROOT) + '/box_image')
        cv2.imwrite(save_path, crop)
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',crop.shape)


def get_images_tensor(path):
    device = 'cuda' if torch.cuda.is_available() else torch.device('cpu')
    tensor_data = []
    image_path_list = os.listdir(path)
    all_image_list = [os.path.join(path,i) for i in image_path_list]
    for i, image_path in enumerate(all_image_list):
        img_read = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        img_read = torch.from_numpy(img_read).float()
        img_read = img_read.to(device)
        tensor_data.append(img_read)
    return tensor_data, image_path_list

# 判断点是否在多边形内的函数
def is_point_in_polygon(point, poly_points):
    """检查点是否在多边形内部或边界上"""
    result = cv2.pointPolygonTest(poly_points, point, False)
    return result >= 0  # 返回值>=0表示在内部或边界上

# RTMPose_utils
def bbox_xyxy2cs_tensor(bbox, padding):
    bbox = torch.from_numpy(bbox)
    dim = bbox.ndim
    if dim ==1:
        bbox = bbox[None,]
    x1, y1, x2, y2 = torch.split(bbox, [1,1,1,1], dim=1)
    center = torch.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = torch.hstack([x2 - x1, y2 - y1]) * padding
    if dim ==1:
        center = center[0]
        scale = scale[0]
    return center, scale
 

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



def prepare_coco_keypoints(keypoints, scores):
    """
    将 (1,17,2) 关键点和 (1,17) 得分合并为 (17,3) 格式
    
    参数:
        keypoints: (1,17,2) 数组
        scores: (1,17) 数组
        
    返回:
        (17,3) 数组，每行是 [x, y, score]
    """
    # 移除批处理维度
    kps = keypoints[0]  # (17,2)
    scs = scores[0]     # (17,)
    
    # 合并坐标和得分
    coco_kps = np.hstack([kps, scs.reshape(-1, 1)])
    return coco_kps


def draw_coco_keypoints(image, keypoints, scores=None, 
                       confidence_threshold=0.2, 
                       keypoint_radius=2, 
                       skeleton_thickness=2):
    """
    绘制COCO格式的17个关键点和骨骼连接（修正版）
    
    参数:
        image: 输入图像 (H,W,3)
        keypoints: (17,2) 或 (17,3) 或 (1,17,2) 数组
        scores: 可选，(17,) 或 (1,17) 数组
        confidence_threshold: 只绘制置信度高于此值的关键点
        keypoint_radius: 关键点圆圈半径
        skeleton_thickness: 骨骼线条粗细
    """
    # 准备关键点数据
    if scores is not None:
        if keypoints.ndim == 3 and keypoints.shape[0] == 1:
            keypoints = keypoints[0]  # (1,17,2) -> (17,2)
        if scores.ndim == 2 and scores.shape[0] == 1:
            scores = scores[0]       # (1,17) -> (17,)
        coco_kps = np.hstack([keypoints, scores.reshape(-1, 1)])
    else:
        if keypoints.ndim == 3 and keypoints.shape[0] == 1:
            keypoints = keypoints[0]  # (1,17,3) -> (17,3)
        coco_kps = keypoints
    
    skeleton = [
        # [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],  # 腿部
        # [5, 11], [6, 12],   # 身体到髋部
        # [4,6],[3,5],
        # [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],  # 手臂
        # [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 0]  # 头部(鼻子到鼻子不画)
        [5, 7], [7, 9], 
        [6, 8], [8, 10]
    ]
    
    # 绘制骨骼连接
    for i, j in skeleton:
        if i < len(coco_kps) and j < len(coco_kps):
            kp1 = coco_kps[i]
            kp2 = coco_kps[j]
            # 跳过鼻子到鼻子的连接
            if i == j:
                continue
            # 检查置信度
            if len(kp1) > 2 and len(kp2) > 2:
                if kp1[2] > confidence_threshold and kp2[2] > confidence_threshold:
                    cv2.line(image, 
                            (int(kp1[0]), int(kp1[1])), 
                            (int(kp2[0]), int(kp2[1])), 
                            (0, 255, 255), skeleton_thickness)
    
    # 绘制关键点
    for i, kp in enumerate(coco_kps):
        if len(kp) == 2:  # 只有xy坐标
            x, y = kp
            conf = 1.0
        else:  # 有置信度
            x, y, conf = kp
        
        if conf > confidence_threshold:
            # 不同部位使用不同颜色
            if i in [0, 1, 2, 3, 4]:  # 头部
                color = (0, 0, 255)  # 红色
            elif i in [5, 6, 7, 8, 9, 10]:  # 上半身
                color = (255, 0, 0)  # 蓝色
            else:  # 下半身
                color = (0, 255, 0)  # 绿色
            
            cv2.circle(image, (int(x), int(y)), keypoint_radius, color, -1)
            # # 添加关键点编号标签
            # cv2.putText(image, str(i), (int(x), int(y)), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image