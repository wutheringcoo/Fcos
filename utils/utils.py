import os
import sys
import cv2 
import time
import math
import torch
import random
import platform
import datetime
import logging
import subprocess
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.distributed as dist

from tqdm import tqdm
from pathlib import Path
from utils.config import Config
logger = logging.getLogger(__name__)

# for training
def seeds(nums):
    random.seed(nums)
    np.random.seed(nums)
    torch.manual_seed(nums)
    torch.cuda.manual_seed(nums)
    torch.cuda.manual_seed_all(nums)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=16):
    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    if os.path.isfile(fname):  # do not overwrite
        return None

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()

    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # Empty array for output
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)

    # Fix class - colour map
    prop_cycle = plt.rcParams['axes.prop_cycle']
    # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    hex2rgb = lambda h: tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    color_lut = [hex2rgb(h) for h in prop_cycle.by_key()['color']]

    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype('int')
            gt = image_targets.shape[1] == 6  # ground truth if no conf column
            conf = None if gt else image_targets[:, 6]  # check for confidence presence (gt vs pred)

            boxes[[0, 2]] *= w
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] *= h
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = color_lut[cls % len(color_lut)]
                cls = names[cls] if names else cls
                if gt or conf[j] > 0.3:  # 0.3 conf thresh
                    label = '%s' % cls if gt else '%s %.1f' % (cls, conf[j])
                    plot_one_box(box, mosaic, label=label, color=color, line_thickness=tl)

        # Draw image filename labels
        if paths is not None:
            label = os.path.basename(paths[i])[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
                        lineType=cv2.LINE_AA)

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    if fname is not None:
        mosaic = cv2.resize(mosaic, (int(ns * w * 0.5), int(ns * h * 0.5)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))

    return mosaic

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

# for dataset
def resize_pad(image,boxes,input_ksize):
        '''
            Functions:
                resize image and bboxes
                1)  将图片的长边和短边resize(800,1333)之间
                    注意区别和联系推理过程中：将图片的长边优先resize到target_length
                        区别：本例中target_length是不相等的两个数
                        联系：都是计算获得最小的放缩因子min(target_length/side1, target_length/side1),另一边padding不起strdie整数倍
                2)  这里padding暴力补齐，在图像的最右侧或者最底册
            Args:
                image       : np.array 
                              shape(H,W,C)
                              dtype=np.uint8 RGB  
                bboxes      : np.array
                              shape(None,4)
            Returns:
                image_paded : np.array
                              shape(800+padding, 1333+padding, C)或者(1333+padding, 800+padding, C) 
                              dtype=np.uint8 RGB 其中padding是可选项 
                bboxes      : shape(None,4)
        '''
        target_short, target_long = input_ksize
        h,  w, _  = image.shape
        short_side = min(w,h)
        long_side  = max(w,h)
        scale1 = target_short / short_side
        scale2 = target_long  / long_side
        scale = min(scale1, scale2)
        nw, nh  = int(round(scale * w)), int(round(scale * h)) 

        # min_side, max_side    = input_ksize
        # h,  w, _  = image.shape
        # smallest_side = min(w,h)
        # largest_side=max(w,h)
        # scale=min_side/smallest_side
        # if largest_side*scale>max_side:
        #     scale=max_side/largest_side
        # nw, nh  = int(scale * w), int(scale * h)
        
        image_resized = cv2.resize(image, (nw, nh))
        pad_w = 32 - nw%32
        pad_h = 32 - nh%32

        image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            # cv2.imwrite('4.jpg', image_paded[:,:,::-1])  # debug
            return image_paded, boxes

# for loss
def femap2ori(feature_map, strides):
    """
        Args:
            feature_map: tensor(batch_size, h, w, nums) nums of class_nums,4,1
        Returns:
            coords: tensor(batch_size*h*w, 2)
        Function:
            Calculate each layer of batch_image' features(pixels) inflection to ori_image
            map_coordx(y) = x(y)*strides + (strides//2)
            exp:
                ori_coord:
                torch.tensor(
                    [[0.,1.],
                    [0.,2.],
                    [1.,0.],
                    [1.,1.],
                    ])

                map_coord: strides = 2
                torch.tensor(
                    [[1.,1.],
                    [1.,3.],
                    [3.,1.],
                    [3.,3.],
                    ])
    """
    h, w = feature_map.shape[1:3]
    x_coord = torch.arange(0, w * strides, strides, dtype = torch.float32, device = feature_map.device)
    y_coord = torch.arange(0, h * strides, strides, dtype = torch.float32, device = feature_map.device) 

    x_grid, y_grid = torch.meshgrid(x_coord, y_coord)
    x_grid = torch.reshape(x_grid, [-1])
    y_grid = torch.reshape(y_grid, [-1])
    coords = torch.stack([x_grid, y_grid], dim = 1) + strides // 2 
    return coords

# for loss
def dist_pix2bound(coord_map, bbox_gt):
    '''
        Args:
            coord_map:
                tensor(batch_size*h*w, 2)
            bbox_gt:
                tensor(batch_size, m', 4) 
        Return:
            tensor(batch_size, feature_h * feature_w, m, 4)
    '''
    x_grid = coord_map[:,0]                             # (feature_h * feature_w,)
    y_grid = coord_map[:,1]                             # (feature_h * feature_w,)
    # (1, feature_h * feature_w, 1) - (batch_size, 1, m') => (batch_size, h*w, m')
    l = x_grid[None,:,None] - bbox_gt[...,0][:,None,:]  
    t = y_grid[None,:,None] - bbox_gt[:,:,1][:,None,:]  # (1, feature_h * feature_w, 1) - (batch_size, 1, m')
    r = bbox_gt[:,:,2][:,None,:] - x_grid[None,:,None] 
    b = bbox_gt[:,:,3][:,None,:] - y_grid[None,:,None] 
    ltrb = torch.stack((l,t,r,b), dim=-1)  
    return ltrb

# for loss
def dist_pix2center(coord_map, bbox_gt):
    '''
        Args:
            coord_map:
                tensor(batch_size*h*w, 2)
            bbox_gt:
                tensor(batch_size, m', 4) 
        Return:
            tensor(batch_size, feature_h * feature_w, m, 4)
    '''
    center_x = (bbox_gt[...,0] + bbox_gt[...,2]) / 2    # (batch_size, m')
    center_y = (bbox_gt[...,1] + bbox_gt[...,3]) / 2    # (batch_size, m')
    x_grid = coord_map[:,0]                             # (feature_h * feature_w,)
    y_grid = coord_map[:,1]                             # (feature_h * feature_w,)
    # (1, feature_h * feature_w, 1) - (batch_size, 1, m') => (batch_size, h*w, m')
    l = x_grid[None,:,None] - center_x[:,None,:]  
    t = y_grid[None,:,None] - center_y[:,None,:]  # (1, feature_h * feature_w, 1) - (batch_size, 1, m')
    r = center_x[:,None,:] - x_grid[None,:,None] 
    b = center_x[:,None,:] - y_grid[None,:,None] 
    ltrb = torch.stack((l,t,r,b), dim=-1)  
    return ltrb

# for loss
def centerness(reg_target):
    """
        Args:
            reg_target: tensor(batch_size, h*w, 4)
        Return:
            centerness: tensor(batch_size, h*w, 1)
    """
    lr_min = torch.min(reg_target[...,0],reg_target[...,2]) # (b, h*w)
    lr_max = torch.max(reg_target[...,0],reg_target[...,2]) # (b, h*w)
    td_min = torch.min(reg_target[...,1],reg_target[...,3])
    td_max = torch.max(reg_target[...,1],reg_target[...,3])
    centerness = (torch.sqrt((lr_min * td_min)/(lr_max * td_max + 1e-10))).unsqueeze(-1)
    return centerness

### for test

def reshape(x):
    '''
        Func: (b,nums_c,h,w) ===> (b, pixels_nums, c_nums)           
    '''
    batchsize = x[0].shape[0]
    c_nums = x[0].shape[1]
    preds = []
    for pred_1layer in x:
        pred_1layer = pred_1layer.permute(0,2,3,1)
        pred_1layer = torch.reshape(pred_1layer, [batchsize, -1, c_nums])
        preds.append(pred_1layer)
    return torch.cat(preds, dim=1)

def ltrb2xyxy(pixels, offset):
    """
        Args:
            pixels:
                (nums_pixels, 2) 
                sums of feature_pixels from 1pic of batch(because each pic in batch have the same nums feature_pixels) 
            offset:
                (b, nums_pixels, 4)
        Returns:
            (b, nums_pixels, 4)         
    """
    x1y1 = pixels[None,:,:] - offset[...,:2]    #(b, nums_pixels, 2)     
    x2y2 = pixels[None,:,:] + offset[...,2:]    #(b, nums_pixels, 2)
    bbox = torch.cat([x1y1, x2y2], dim = -1)    #(b, nums_pixels, 4)
    return bbox

def non_max_suppression(out, conf_thresh = 0.05, iou_thres = 0.6):
    cls_prob, cls_class, reg_preds = out
    batch_size = cls_prob.shape[0]
    topk_threshold = min(Config.max_pixels, cls_prob.shape[-1])
    _, prob_ind_topk = torch.topk(cls_prob, topk_threshold, dim = -1, largest = True, sorted = True) #prob_ind_topk:(b, topk_pixels_nums)
    cls_prob_1pic = []
    cls_class_1pic = []
    reg_preds_1pic = []
    for i in range(batch_size):
        cls_prob_1pic.append(cls_prob[i][prob_ind_topk[i]])     #[(topk_pixels_nums,),(topk_pixels_nums,)... ]
        cls_class_1pic.append(cls_class[i][prob_ind_topk[i]])   #[(topk_pixels_nums,),(topk_pixels_nums,)... ]
        reg_preds_1pic.append(reg_preds[i][prob_ind_topk[i]])   #[(topk_pixels_nums,4),(topk_pixels_nums,4)... ] 
    cls_prob_topk = torch.stack(cls_prob_1pic, dim=0)           #(b, topk_pixels_nums)
    cls_class_topk = torch.stack(cls_class_1pic, 0)             #(b, topk_pixels_nums)
    reg_preds_topk = torch.stack(reg_preds_1pic, 0)             #(b, topk_pixels_nums,4)

    cls_prob_score_post = []
    cls_class_score_post = []
    reg_preds_score_post = []
    
    for i in range(batch_size):
        cls_score_mask = cls_prob_topk[i] >= conf_thresh     #maybe topk_pixels_nums1 != topk_pixels_nums2
        cls_prob_score = cls_prob_topk[i][cls_score_mask]    #(topk_pixels_nums, )
        cls_class_score = cls_class_topk[i][cls_score_mask]  #(topk_pixels_nums, )
        reg_preds_score = reg_preds_topk[i][cls_score_mask]  #(topk_pixels_nums,4)

        # nms_keep = nms(reg_preds_score, cls_prob_score, cls_class_score, iou_thres) # method2: 
        nms_keep = nms_api(reg_preds_score, cls_prob_score, cls_class_score, iou_thres) # speed_mehtod1 is method2's 352
        
        cls_prob_score_post.append(cls_prob_score[nms_keep])     #[(nms_keep1, ), (nms_keep2, )]
        cls_class_score_post.append(cls_class_score[nms_keep])   #[(nms_keep1, ), (nms_keep2, )]
        reg_preds_score_post.append(reg_preds_score[nms_keep])   #[(nms_keep1,4), (nms_keep2,4)]
    return cls_prob_score_post, cls_class_score_post, reg_preds_score_post    

def nms(bbox, bbox_score, class_ind, iou_threshold):
    """
        Args:
            bbox: tensor(nums_after_cls_thred, 4)
            bbox_score: (topk_pixels_nums, )
        Returns:
            tensor(nums_nms_left, ) dtype = int64
    """
    if bbox.shape[0] == 0:
        return torch.zeros(0, device = bbox.device).long()
    bbox = bbox + 10000*class_ind.unsqueeze(-1)
    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]
    area = (x2-x1) * (y2-y1)
    bbox_nums = bbox.shape[0]
    nms_keep = []
    order = torch.argsort(bbox_score, descending= True)
    # _, order = torch.sort(bbox_score)
    # order = torch.arange(bbox_score.shape[0], device = bbox.device)
    if order.numel() == 1:
        return torch.tensor([0])
    while order.numel() >= 1:
        if order.numel() == 1:
            nms_keep.append(order.item())
            break
        nms_keep.append(order[0].item())
        xmin = torch.max(x1[order[1:]], x1[order[0]])   #((m-1, ) compare with (1,)) ===> (m-1, )
        ymin = torch.max(y1[order[1:]], y1[order[0]])
        xmax = torch.min(x2[order[1:]], x2[order[0]])
        ymax = torch.min(y2[order[1:]], y2[order[0]])
        inter = (xmax - xmin).clamp(min=0) * (ymax - ymin).clamp(min=0)
        iou = inter / (area[0] + area[order[1:]] - inter)
        mask = iou <= iou_threshold
        if mask.numel() == 0:
            break
        order = order[1:][mask]
        #method2: idx = (iou <=iou_threshold).nonzero().squeeze()
    return torch.LongTensor(nms_keep).to(bbox.device) #list => tensor is placed in cpu

def nms_api(bbox, bbox_score, class_ind, iou_threshold):
    import torchvision
    nms_out_index = torchvision.ops.batched_nms(
            bbox,                     #(1000,4)
            bbox_score,               #(1000,)
            class_ind,                #(1000,)
            iou_threshold,
        )
    return nms_out_index

def descend_sort(conf_pred, class_pred, bbox_pred):
    """
    Returns:
        numpy
    Function:
        make sure results of preds/pic descending order
    """
    descend_ind= [(-score).argsort() for index, score in enumerate(conf_pred)]
    conf_pred  = [sample_boxes[mask].cpu().numpy() for sample_boxes, mask in zip(conf_pred,  descend_ind)]
    class_pred = [sample_boxes[mask].cpu().numpy()  for sample_boxes, mask in zip(class_pred, descend_ind)]
    bbox_pred  = [sample_boxes[mask].cpu().numpy() for sample_boxes, mask in zip(bbox_pred,  descend_ind)]
    return conf_pred, class_pred, bbox_pred

def iou_bbox(bbox1, bbox2):
    """
    Args:np.array
        bbox1 : n x 4
        bbox2 : m x 4
    """
    area1 = (bbox1[:,2] - bbox1[:,0]) * (bbox1[:,3] - bbox1[:,1])    #(n, )
    area2 = (bbox2[:,2] - bbox2[:,0]) * (bbox2[:,3] - bbox2[:,1])    #(m, )
    xmin = np.maximum(bbox1[:,0][:,None] , bbox2[:,0][None,:])  #(n,m)
    ymin = np.maximum(bbox1[:,1][:,None] , bbox2[:,1][None,:])  #(n,m)
    xmax = np.minimum(bbox1[:,2][:,None] , bbox2[:,2][None,:])  #(n,m)
    ymax = np.minimum(bbox1[:,3][:,None] , bbox2[:,3][None,:])  #(n,m)
    intersection = np.maximum(0, (xmax - xmin)) *  np.maximum(0, (ymax - ymin))  #(n,m)
    iou = intersection / (area1[:,None] + area2[None,:] - intersection) 
    return iou

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list/numpy).
        precision: The precision curve (list/numpy).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def ap_metrics(bbox_gt, class_gt, conf_pred, class_pred, bbox_pred, class_nums, iou_thread):
    """
    Attention:
        confidence has been sorted after nms 
    Arguments:
        conf_pred:  tensor in list   #每张图片中的所有预测框降序排列，此处不变量是图片id，所以后期遍历所有图片的当个类别时候，对应类别的置信度需要再次降序排列
            [(a, ), (b, ), (c, )...] #a代表保留的预测框数量(nms保留下的anchor point)，数值为置信度，已排序
        class_pred: tensor in list
            [(a, ), (b, ), (c, )...]  sparse label 
        bbox_pred:  tensor in list
            [(a, 4), (b, 4), (c, 4)...]
        class_nums: 
            20 + 1 (COCO:80+1)
    Returns: 
        a dict containing average precision for each cls
    """
    all_ap = {}
    for label in tqdm(range(1, class_nums)):
        class_gt_mask  = [cls_gt == label for cls_gt in class_gt]
        bbox_gt_1class = [bbox[mask] for bbox, mask in zip(bbox_gt, class_gt_mask)]

        class_pred_mask  = [cls_pred == label for cls_pred in class_pred]
        bbox_pred_1class = [bbox[mask] for bbox, mask in zip(bbox_pred, class_pred_mask)]
        conf_pred_1class = [conf[mask] for conf, mask in zip(conf_pred, class_pred_mask)]

        total_gts = 0
        tp, fp, scores = np.zeros((0, )), np.zeros((0, )), np.zeros((0, ))
        # for each pic
        for bbox_gt_1pic, bbox_pred_1pic, conf_pred_1pic in zip(bbox_gt_1class, bbox_pred_1class, conf_pred_1class):
            total_gts = total_gts + bbox_gt_1pic.shape[0]
            assigned_gt = []  
            #for 1sample of 1pic
            for index in range(bbox_pred_1pic.shape[0]): 
                scores = np.append(scores, conf_pred_1pic[index])  #note all pics’ confidence
                if len(bbox_gt_1pic) == 0:  # if no gts found for the predicted bbox, assign the bbox to fp
                    tp = np.append(tp, 0)
                    fp = np.append(fp, 1)
                    continue
                bbox_preds = bbox_pred_1pic[index][None,:]    #(1,4)
                iou = iou_bbox(bbox_gt_1pic, bbox_preds)       #[(m,4),(1,4)] ==> (m,1) 
                max_iou_ind = np.argmax(iou, axis=0)           #(1, ) array
                max_iou = iou[max_iou_ind, 0]
                if max_iou >= iou_thread and max_iou_ind not in assigned_gt:
                    fp = np.append(fp, 0)
                    tp = np.append(tp, 1)
                    assigned_gt.append(max_iou_ind)
                else:
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
        # sort by score
        indices = np.argsort(-scores)  #单个类别中的所有图片的所有预测框降序排列(必不可少)
        fp = fp[indices]
        tp = tp[indices]
        # compute cumulative false positives and true positives
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        # compute recall and precision
        recall = tp / (total_gts + 1e-16)                                 #(nums_bbox_pred_1cls, )
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)    #(nums_bbox_pred_1cls, )
        ap = compute_ap(recall, precision)    
        all_ap[label] = ap
    return all_ap

# for model
def clipbbox(batch_images, batch_bbox):
    """
    Args:
        batch_images: (b, c, h, w)
        batch_boxes: (b, nums_pixels, 4)
    """
    batch_bbox.clamp_(min=0)
    h, w = batch_images.shape[2:]
    batch_bbox[...,[0,2]].clamp_(max=w-1)
    batch_bbox[...,[1,3]].clamp_(max=h-1)
    return batch_bbox

# genera_paras 
def DDP_launch(opt):
    '''
        1 : set GPU id to each process(local_rank)
        2 : initialization[backend, env]
    '''
    assert torch.cuda.device_count() > opt.local_rank
    # torch.cuda.set_device(local_rank) = CUDA_VISIBLE_DEVICES=local_rank
    torch.cuda.set_device(opt.local_rank)           # torch.cuda.set_device(local_rank+2): point 3rd and 4th GPU for 1st(0) and 2nd(1)process 
    device = torch.device('cuda', opt.local_rank)   # set GPU id to process
    # print('device', device, local_rank ) # debug correlation betwee process with gpu id
    if opt.world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')   # distributed backend
    assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
    opt.batch_size = opt.total_batch_size // opt.world_size
    return device
    
def set_logging(rank=-1):
        logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)

def select_device(device='', apex=False, batch_size=None):
    # device = 'cpu' or '0' or id in '0,1,2,3'
    s = f'FCOS 🚀 {git_describe() or date_modified()} torch {torch.__version__} '  # string
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA ' + ('Apex ' if apex else '')  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            logger.info("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / c))
    else:
        logger.info('Using CPU')

    logger.info('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')
    # if cpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    # elif device:  # non-cpu device requested
    #     os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
    #     assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    # cuda = not cpu and torch.cuda.is_available()
    # if cuda:
    #     n = torch.cuda.device_count()
    #     if n > 1 and batch_size:  # check that batch_size is compatible with device_count
    #         assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
    #     space = ' ' * len(s)
    #     for i, d in enumerate(device.split(',') if device else range(n)):
    #         p = torch.cuda.get_device_properties(i)
    #         s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    # else:
    #     s += 'CPU\n'

    # logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    # return torch.device('cuda:0' if cuda else 'cpu')

def git_describe(path=Path(__file__).parent):  # path must be a directory
    # return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e:
        return ''  # not a git repository

def date_modified(path=__file__):
    # return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'
