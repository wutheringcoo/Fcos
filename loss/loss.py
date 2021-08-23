import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import Config
from utils.utils import femap2ori, dist_pix2bound, dist_pix2center, centerness

class Targets_build(nn.Module):
    def __init__(self, strides, window, sample_ratio):
        super(Targets_build, self).__init__()
        self.strides = strides
        self.window = window
        self.sample_ratio = sample_ratio

    def forward(self, preds, bbox_gt, cls_gt):
        '''
            Args: 
                preds   : tuple(cls_head, reg_head, cnt_head)
                          cls_head [tensor1,tensor2,tensor3,tensor4,tensor5]
                          reg_head [tensor1,tensor2,tensor3,tensor4,tensor5]
                          cnt_head [tensor1,tensor2,tensor3,tensor4,tensor5]
                bbox_gt : torch.Size([batch_size, nums_boxes, 4])
                cls_gt  : torch.Size([batch_size, nums_boxes])
            Returns:
                cls_target :  [tensor1,tensor2,tensor3,tensor4,tensor5] (batch_size,h*w,1)
                reg_target :  [tensor1,tensor2,tensor3,tensor4,tensor5] (batch_size,h*w,4)
                cnt_target :  [tensor1,tensor2,tensor3,tensor4,tensor5] (batch_size,h*w,1)
        '''
        cls_target, reg_target, cnt_target = [], [], []
        for i, preds_layer in  enumerate(zip(*preds)):
            targets_layer = self.targets_build_layer(preds_layer, bbox_gt, cls_gt, self.strides[i], self.window[i], self.sample_ratio)
            cls_target.append(targets_layer[0])
            reg_target.append(targets_layer[1])
            cnt_target.append(targets_layer[2])
        return cls_target, reg_target, cnt_target

    def targets_build_layer(self, preds_layer, bbox_gt, cls_gt, strides, window, sample_ratio):
        ''' 
            Functions:
                    Fcos detector directly views locations as training samples 
                instead of anchor boxes in anchor-based detectors, which is the 
                same as FCNs for semantic segmentation 
            Args:
                out_layer:
                    tuple(cls_head, reg_head, cnt_head) from one layer of head
                    cls_head_layer: tensor(batch_size, nums_class, h, w)
                    reg_head_layer: tensor(batch_size, 4, h, w) with (l,t,r,b)
                    cnt_head_layer: tensor(batch_size, 1, h, w)
                bbox_gt:
                    tensor shape(batch_size, m, 4)  m denotes nums of bbox_gt
                cls_gt:
                    tensor shape(batch_size, m)
            Returns:
                cls_target           # [batch_size,h*w,1]
                reg_target           # [batch_size,h*w,4]
                cnt_target           # [batch_size,h*w,1]
        '''
        cls_head_layer, _, _ = preds_layer
        cls_head_layer = cls_head_layer.permute(0,2,3,1)
        batch_size = cls_head_layer.shape[0]
        
        #1 feature map_coords(mapped to ori img) and dist 
        coord_map = femap2ori(cls_head_layer, strides)
        dist_bound = dist_pix2bound(coord_map.to(device = bbox_gt.device), bbox_gt)
        dist_center = dist_pix2center(coord_map.to(device = bbox_gt.device), bbox_gt)
        radius = strides * sample_ratio  # (12,24,48,96,192)
        
        #2 positive samples trciks: 
        """    1)-> 2) -> 3) -> 4) mask_order of p_sampling
            1) pixel locates in box_gt
            2) pixel in ith feat_layer executes cls and reg tasks conditonal on: 
                dist_max_i-1 <= max(l_i,t_i,r_i,b_i) <= dist_max_i 
            3) pixel center sampling, maxdis_pix2center in circle region of center_bbox_gt
            4) pixel only mapping to one bbox_gt(min area) [one pixel to one bbox_gt]
        """
        dist_bound_min, _ = torch.min(dist_bound, dim = -1)  # (b,h*w,m,4) => (b,h*w,m)
        dist_bound_max, _ = torch.max(dist_bound, dim = -1)  # (b,h*w,m,4) => (b,h*w,m)
        dist_center_max, _ = torch.max(dist_center, dim = -1)# (b,h*w,m,4) => (b,h*w,m)
        pos_mask = (dist_bound_min > 0)&(dist_bound_max <= window[1])&(dist_bound_max >= window[0])\
                   &(dist_center_max < radius)               # [b,h*w,m]
        
        areas_bbox = (dist_bound[...,0] + dist_bound[...,2]) * (dist_bound[...,1] + dist_bound[...,3])                 # [b,h*w,m]
        areas_bbox[~pos_mask] = 99999999                    # [b,h*w,m]
        _, areas_min_ind = torch.min(areas_bbox, dim=-1)    # [b,h*w]
        mask_areas_min = torch.zeros_like(areas_bbox, dtype = torch.bool).scatter_(-1, areas_min_ind.unsqueeze(-1), 1) # [b,h*w,m]
        
        #3 target mapping according to (mask_areas_min and pos_mask)
        cls_target_layer, _ = torch.broadcast_tensors(cls_gt.clone()[:,None,:], areas_bbox) # [b,h*w,m]
        # method 2
        # cls_targets=cls_target_layer[torch.zeros_like(areas_bbox,dtype=torch.bool).scatter_(-1,areas_min_ind.unsqueeze(dim=-1),1)]
        # cls_targets=torch.reshape(cls_targets,(batch_size,-1,1))
        cls_target_layer[~mask_areas_min] = 0                           # [b,h*w,m]
        cls_target_layer = cls_target_layer.sum(-1).unsqueeze(-1)       # [b,h*w,1]
        # print((cls_target_layer == cls_targets).sum())
        reg_target_layer = dist_bound[mask_areas_min]                   # [b*h*w,4]
        reg_target_layer = reg_target_layer.reshape(batch_size, -1, 4)  # [b,h*w,4]
        cnt_target_layer = centerness(reg_target_layer)                 # [b,h*w,1]
        #Attention : all pixel have one mapping bbox_gt，filter not fit to 1&2&3 conditions pixel.
        pos_mask_ = pos_mask.long().sum(dim=-1)                   # [b,h*w]
        pos_mask_ = pos_mask_ >= 1                                # [b,h*w]
        cls_target_layer[~pos_mask_] = 0                          # [batch_size,h*w,1]
        reg_target_layer[~pos_mask_] = -1.                        # [batch_size,h*w,4]
        cnt_target_layer[~pos_mask_] = -1.                        # [batch_size,h*w,1]
        return cls_target_layer, reg_target_layer, cnt_target_layer

class Loss_Compute(nn.Module):
    def __init__(self):
        super(Loss_Compute, self).__init__()
        self.config = Config
        self.targets_building = Targets_build(self.config.strides, self.config.window, self.config.sample_ratio)
        
    def forward(self, preds, targets, only_pos_calculate = Config.pos_sample_cls):
        """
            Args:
                preds   : tuple(cls_head, reg_head, cnt_head)
                          cls_head [tensor1,tensor2,tensor3,tensor4,tensor5] (batch_size,cls_nums,h,w)
                          reg_head [tensor1,tensor2,tensor3,tensor4,tensor5] (batch_size,4,h,w)
                          cnt_head [tensor1,tensor2,tensor3,tensor4,tensor5] (batch_size,1,h,w)
                targets : bbox_gt, cls_gt
                          bbox_gt     : tensor torch.Size([batch_size, nums_boxes, 4])
                          cls_gt      : tensor torch.Size([batch_size, nums_boxes])
            Returns:
                cls_loss : tensor(1,) loss_mean/pixel/pic
                reg_loss : tensor(1,) loss_mean/pixel/pic
                cnt_loss : tensor(1,) loss_mean/pixel/pic
        """ 
        self.b = preds[0][0].shape[0]  # batch_size
        self.n = preds[0][0].shape[1]  # nums of class
        cls_preds, reg_preds, cnt_preds = [], [], []
        for i , pred_sub in enumerate(preds):
            for pred in pred_sub:
                pred = pred.permute(0, 2, 3, 1)
                if i == 0:
                    pred = pred.reshape(self.b, -1, self.n)
                    cls_preds.append(pred)
                elif i == 1:
                    pred = pred.reshape(self.b, -1 , 4)
                    reg_preds.append(pred)
                elif i == 2:
                    pred = pred.reshape(self.b, -1 , 1)
                    cnt_preds.append(pred)
        cls_preds = torch.cat(cls_preds,dim=1)           #(b, h*w*5, cls_nums)
        reg_preds = torch.cat(reg_preds,dim=1)           #(b, h*w*5, 4)
        cnt_preds = torch.cat(cnt_preds,dim=1)           #(b, h*w*5, 1)
        # import os
        # rank =int(os.environ['RANK'])
        # if rank == 0:
        #     print('targets', targets)
         
        cls_targets, reg_targets, cnt_targets = self.targets_building(preds, targets[0], targets[1])
        cls_targets = torch.cat(cls_targets,dim=1)       #(b, h*w*5, 1)
        reg_targets = torch.cat(reg_targets,dim=1)       #(b, h*w*5, 4)
        cnt_targets = torch.cat(cnt_targets,dim=1)       #(b, h*w*5, 1)
        
        # assert cls_targets.shape==cls_preds.shape
        assert reg_targets.shape==reg_preds.shape
        assert cnt_targets.shape==cnt_preds.shape
        
        pos_masks = cnt_targets > -1   #(b, h*w*5, 1)bool
        if only_pos_calculate:
            cls_loss = self.compute_cls_loss(cls_preds, cls_targets, only_pos_calculate, pos_masks)
        else:
            cls_loss = self.cls_compute(cls_preds, cls_targets, only_pos_calculate, pos_masks)  ############
        reg_loss = self.reg_compute(reg_preds, reg_targets, pos_masks)
        cnt_loss = self.cnt_compute(cnt_preds, cnt_targets, pos_masks)
        loss = (cls_loss+reg_loss+cnt_loss)*self.b  ############
        return loss, (cls_loss.detach(), reg_loss.detach(), cnt_loss.detach(), loss.detach())   ############

    def cls_compute(self, cls_preds,cls_targets,only_pos_calculate, pos_masks):
        """
        1)先计算每张图的loss，
        2)将每张图的loss / 所有正样本像素总和
        Args:
            cls_preds   : #(b, h*w*5, cls_nums)
            cls_targets : #(b, h*w*5, 1)
        Returns:
            tensor (1,)
        """
        pos_pixels = cls_targets[0].shape[0] * torch.ones(self.b, device = cls_targets.device)   #(b,) all pixels calculated for each picture
        cls_loss_picture = []
        for i in range(self.b):
            if only_pos_calculate:
                cls_targets_1pic = (torch.arange(1, self.n+1, device = cls_targets.device)[None,:] == cls_targets[i]).float()  #(h*w*5, cls_num)
                pos_pixels = torch.sum(pos_masks,dim=(1,2)).float()                 #(b,) pos for each picture
            else:
                cls_targets_1pic = torch.zeros(cls_targets[i].shape[0],self.n, device = cls_targets.device).scatter_(dim = -1, index=cls_targets[i].long(), value=1)  #(h*w*5, cls_num)
            cls_loss_picture.append(self.focal_loss(cls_preds[i],cls_targets_1pic))
        cls_loss_pixel = torch.cat(cls_loss_picture, dim=-1) / pos_pixels           #(b,) loss/pix/pic
        return cls_loss_pixel.mean().unsqueeze(0)

    def compute_cls_loss(self, preds, targets, only_pos_calculate, pos_masks):
        '''
        Args  
        preds: list contains five level pred [batch_size,class_num,_h,_w]
        targets: [batch_size,sum(_h*_w),1]
        mask: [batch_size,sum(_h*_w),1]
        '''
        batch_size = targets.shape[0]
        preds_reshape = []
        class_num = preds[0].shape[1] 
        pox_pixels = torch.sum(pos_masks,dim=[1,2]).clamp_(min=1).float() #[batch_size,]
        assert preds.shape[:2]==targets.shape[:2]
        loss=[]
        for batch_index in range(batch_size):
            pred_pos = preds[batch_index]       #[sum(_h*_w),class_num]
            target_pos = targets[batch_index]   #[sum(_h*_w),1]
            target_pos=(torch.arange(1,class_num+1,device=target_pos.device)[None,:]==target_pos).float()#sparse-->onehot
            loss.append(focal_loss_from_logits(pred_pos,target_pos).view(1)) #[(1, ),(1, ), ...] len=batch_size
        cls_loss_pixel = torch.cat(loss,dim=0) / pox_pixels                  #(b, ) len=batch_size
        return cls_loss_pixel.mean().unsqueeze(0) # [batch_size,]

    def reg_compute(self, reg_preds, reg_targets, pos_masks):
        pos_pixels = torch.sum(pos_masks,dim=(1,2)).clamp_(min=1).float()                  #(b, )
        reg_loss_picture = []
        for i in range(self.b):
            reg_preds_pos = reg_preds[i][pos_masks.squeeze(-1)[i]]           #(pos_nums, 4)
            reg_targets_pos = reg_targets[i][pos_masks.squeeze(-1)[i]]       #(pos_nums, 4)
            reg_loss_picture.append(self.iou_loss(reg_preds_pos, reg_targets_pos))
        reg_loss_pixel = torch.cat(reg_loss_picture, dim=-1) / pos_pixels    #(b, )
        return reg_loss_pixel.mean().unsqueeze(0)

    def cnt_compute(self, cnt_preds, cnt_targets, pos_masks):
        pos_pixels = torch.sum(pos_masks,dim=(1,2)).clamp_(min=1).float()                  #(b, )
        cnt_loss_picture = []
        for i in range(self.b):
            cnt_preds_pos = cnt_preds[i][pos_masks.squeeze(-1)[i]]           #(pos_nums, 4)
            cnt_targets_pos = cnt_targets[i][pos_masks.squeeze(-1)[i]]       #(pos_nums, 4)
            cnt_loss_picture.append(self.bce_loss(cnt_preds_pos, cnt_targets_pos))
        cnt_loss_pixel = torch.cat(cnt_loss_picture, dim=-1)/pos_pixels      #(b, )
        return cnt_loss_pixel.mean().unsqueeze(0)

    def focal_loss(self, preds, targets, alpha=0.25, gamma=2.0):
        """
            Arguments:
                preds   (h*w*5, cls_num)
                targets (h*w*5, cls_num)
            Returns:
                loss    tensor(1, )     
        """
        preds = preds.sigmoid()
        p_t = preds * targets + (1-preds) * (1.0-targets)
        a_t = alpha * targets + (1-alpha) * (1.0-targets)
        loss = -a_t * torch.pow((1.0-p_t), gamma) * p_t.log()
        return loss.sum().reshape(1)
    
    def iou_loss(self, reg_preds_pos, reg_targets_pos, GIOU=False, DIOU=False, CIOU=True, eps=1e-7):
        """
            Args:
                reg_preds_pos (pos_nums, 4)
                reg_targets   (pos_nums, 4)
            Returns:
                loss_iou      (1, )

            Attention:
                GIOU : [-1,1] -1 when without overlap and far away / 1 when coincide
                CIOU : [-1,1] -1 when without overlap and far away / 1 when coincide  
        """
        w_o = torch.min(reg_preds_pos[:,0], reg_targets_pos[:,0]) + torch.min(reg_preds_pos[:,2], reg_targets_pos[:,2])
        h_o = torch.min(reg_preds_pos[:,1], reg_targets_pos[:,1]) + torch.min(reg_preds_pos[:,3], reg_targets_pos[:,3])
        overlap = w_o * h_o
        
        w_preds = reg_preds_pos[:,0] + reg_preds_pos[:,2]
        h_preds = reg_preds_pos[:,1] + reg_preds_pos[:,3]
        w_targets = reg_targets_pos[:,0] + reg_targets_pos[:,2]
        h_targets = reg_targets_pos[:,1] + reg_targets_pos[:,3]
        area_pred = w_preds * h_preds
        area_target = w_targets * h_targets
        union = area_pred + area_target - overlap
        iou = overlap / union
        
        C_w = w_preds + w_targets - w_o
        C_h = h_preds + h_targets - h_o  
        if GIOU:
            C = C_w * C_h
            loss_iou = 1 - (iou - (C - union)/C)
        elif DIOU or CIOU:
            c_w = w_preds/2 + w_targets/2 - w_o
            c_h = h_preds/2 + h_targets/2 - h_o
            dis_diag1 = torch.pow(c_w,2) + torch.pow(c_h,2)
            dis_diag2 = torch.pow(C_w,2) + torch.pow(C_h,2)
            if DIOU:
                loss_iou = 1- (iou - dis_diag1/dis_diag2)
            elif CIOU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * torch.pow( torch.atan(w_preds/h_preds)-torch.atan(w_targets/h_targets), 2) 
                with torch.no_grad():
                    alpha = v / (1-iou+v+eps)
                loss_iou = 1 - (iou - (dis_diag1/dis_diag2 + alpha * v))
        return loss_iou.sum().reshape(1)
        
        # method2
        # lt_min = torch.min(reg_preds_pos[:,:2], reg_targets_pos[:,:2])
        # rb_min = torch.min(reg_preds_pos[:,2:], reg_targets_pos[:,2:])
        # wh_min = lt_min + rb_min
        # overlap_ = wh_min[:,0] * wh_min[:,1]
        # area1 = (reg_preds_pos[:,0] + reg_preds_pos[:,2]) * (reg_preds_pos[:,1] + reg_preds_pos[:,3])
        # area2 = (reg_targets_pos[:,0] + reg_targets_pos[:,2]) * (reg_targets_pos[:,1] + reg_targets_pos[:,3])
        # union_ = area1 + area2 - overlap_
        # iou_ = overlap_/ union_
        # lt_max = torch.max(reg_preds_pos[:,:2], reg_targets_pos[:,:2])
        # rb_max = torch.max(reg_preds_pos[:,2:], reg_targets_pos[:,2:])
        # wh_max = lt_max + rb_max
        # C_ = wh_max[:,0] * wh_max[:,1]
        # giou = iou - (C_-union)/C_
    
    def bce_loss(self, cnt_preds_pos, cnt_targets_pos):
        loss_bce = F.binary_cross_entropy_with_logits(cnt_preds_pos, cnt_targets_pos, reduction= 'sum')
        
        # method2
        # cnt_preds_pos = cnt_preds_pos.sigmoid()
        # loss_= -(cnt_targets_pos*cnt_preds_pos.log()+(1-cnt_targets_pos)*(1-cnt_preds_pos).log()).mean()
        # print(loss_bce - loss_)
        
        return loss_bce.reshape(1)

def focal_loss_from_logits(preds,targets,gamma=2.0,alpha=0.25):
    '''
    Args:
    preds: [n,class_num] 
    targets: [n,class_num]
    Returns:

    '''
    preds=preds.sigmoid()
    pt=preds*targets+(1.0-preds)*(1.0-targets)
    w=alpha*targets+(1.0-alpha)*(1.0-targets)
    loss=-w*torch.pow((1.0-pt),gamma)*pt.log()
    return loss.sum()
        


        










    




    
