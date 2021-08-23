import sys
import time
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from voc.voc import voc_ap_metrics, convert_to_voc_format
from utils.config import Config
from collections import ChainMap
from utils.dist import time_synchronized
from utils.utils import non_max_suppression, ap_metrics

def test(model, test_dataloader, distributed=False):
    """"
        Args:preds
            (cls_head, reg_head, cnt_head) tuple
            cls_head: 
                    list[tensor1,...]  len=5(layers feature_maps)
                    where  tensor1(Batch_size, nums_class, h, w)          
            reg_head: list[tensor1,...] tensor1(Batch_size, 4, h, w) with (l,t,r,b)
            cnt_head: list[tensor1,...] tensor1(Batch_size, 1, h, w)
        Returns:
            ap: dict
    """
    data_list = {}
    mAP, nms_time, forward_time = 0., 0., 0.
    bbox_gt, class_gt = [], []
    conf_pred, class_pred, bbox_pred = [],[],[]
    device = next(model.parameters()).device
    model.eval()
    for index, (imgs, bbox_targets, classes_targets, paths, img_id) in enumerate(tqdm(test_dataloader)):
      
        imgs, bbox_targets, classes_targets = imgs.to(device, non_blocking = True),bbox_targets.to(device), classes_targets.to(device)
        with torch.no_grad():
            is_time_record = index < len(test_dataloader) - 1 # skip the the last iters since batchsize might be not enough for batch inference
            if is_time_record:
                start = time.time()
                
            # Run model
            out = model(imgs, bbox_targets, classes_targets)
            if is_time_record:
                forward_end = time_synchronized()
                forward_time += forward_end - start

            # Run NMS
            out = non_max_suppression(out, conf_thresh = Config.cls_score_threshold, iou_thres = Config.iou_threshold)
            if is_time_record:
                nms_end = time_synchronized()
                nms_time += nms_end - forward_end
            
            # method2 传入列表的元素位于GPU中，还得再循环一次放入CPU中
            # conf_pred  += out[0]    
            # class_pred += out[1]    
            # bbox_pred  += out[2]    
            
            conf_pred.append(out[0][0].cpu().numpy())     #len = nums_test, [(nms_keep1,),(nms_keep1,),...,(nms_keep2,),(nms_keep2,),...]
            class_pred.append(out[1][0].cpu().numpy())    #len = nums_test, [(nms_keep1,),(nms_keep1,),...,(nms_keep2,),(nms_keep2,),...] 
            bbox_pred.append(out[2][0].cpu().numpy())     #len = nums_test, [(nms_keep1,),(nms_keep1,),...,(nms_keep2,),(nms_keep2,),...]            

        for i in range(imgs.shape[0]):
            bbox_gt.append(bbox_targets[i].cpu().numpy())         #len = batch_size, [(nums, 4), (nums, 4), (nums, 1)]
            class_gt.append(classes_targets[i].cpu().numpy())     #len = batch_size, [(nums, ), (nums ,), (nums, )]
        
        data_list.update(convert_to_voc_format(conf_pred, class_pred, bbox_pred, img_id))
    # pred_conf, pred_class, pred_bbox = descend_sort(conf_pred, class_pred, bbox_pred)  #tensor => numpy #method2排序次数多了一次，其次多一次循环用于转移到cpu中
    # ap = ap_metrics(bbox_gt, class_gt, pred_conf, pred_class, pred_bbox, Config.class_nums, 0.5)
    
  
    # mAP50, mAP5095 = voc_ap_metrics(data_list, forward_time, nms_time, test_dataloader, distributed = False)
    # return mAP50, mAP5095

    ap = ap_metrics(bbox_gt, class_gt, conf_pred, class_pred, bbox_pred, Config.class_nums, 0.5)
    for class_id, class_mAP in ap.items():
        mAP += float(class_mAP)
    mAP /= (Config.class_nums-1)
    return ap, mAP
    





    
    


