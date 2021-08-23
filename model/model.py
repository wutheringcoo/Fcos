import sys
import torch
import torch.nn as nn

from loss.loss import Loss_Compute
from model.backbone import resnet50
from model.neck import Fpn
from model.head import Fcos_head
from utils.config import Config
from utils.utils import femap2ori, reshape, ltrb2xyxy, clipbbox

class Fcos(nn.Module):
    def __init__(self):
        super(Fcos, self).__init__()
        self.config = Config
        self.backbone = resnet50(pretrained=True, weights = self.config.resnet50_weights, num_classes =self.config.num_classes)
        self.fpn = Fpn(features = self.config.features)
        self.head = Fcos_head(features = self.config.features, num_classes = self.config.num_classes)
        
    # def train(self, mode=True):
    #     '''
    #     set module training mode, and frozen bn
    #     '''   
    #     super().train(mode=True)
    #     def freeze_bn(module):
    #         if isinstance(module,nn.BatchNorm2d):
    #             module.eval()
    #         classname = module.__class__.__name__
    #         if classname.find('BatchNorm') != -1:
    #             for p in module.parameters(): p.requires_grad=False
    #     if self.config.freeze_bn:
    #         self.apply(freeze_bn)
    #     if self.config.freeze_stage_1:
    #         self.backbone.freeze_stages(2)
   
    def forward(self, imgs):
        '''
            Args:
                imgs from dataloader
        '''
        all_p = self.fpn(self.backbone(imgs))
        preds = self.head(all_p)
        return preds
  
class Detector(nn.Module):
    def __init__(self):
        super(Detector,self).__init__()
        self.fcos = Fcos()
        self.loss = Loss_Compute()

    def forward(self, imgs, bbox_gt, cls_gt, only_pos_calculate = Config.pos_sample_cls):
        '''
            Args:
                imgs, bbox_gt, cls_gt from dataloader
        '''
        preds = self.fcos(imgs)
        targets = (bbox_gt, cls_gt)
        if self.training:
            loss_train = self.loss(preds, targets)
        else:
            cls_head, reg_head, cnt_head = preds
            pixels_reflect_1layer = []
            for i, stride in enumerate(Config.strides):
                pixels_reflect_1layer.append(femap2ori(cls_head[i].permute(0,2,3,1), stride))
            pixels_reflect = torch.cat(pixels_reflect_1layer, dim=0)        #(nums_pixels, 2)
            
            cls_preds = reshape(cls_head).sigmoid()                         #(b, nums_pixels, classes_nums)
            cnt_preds = reshape(cnt_head).sigmoid()                         #(b, nums_pixels, 1)
            cls_prob, cls_class = torch.max(cls_preds, dim = -1)            #(b, nums_pixels)
            if only_pos_calculate:
                cls_class = cls_class + 1                                   #(b, nums_pixels)
            if Config.joint_centerness_prob:
                cls_prob = torch.sqrt(cls_prob * cnt_preds.squeeze(-1))     #(b, nums_pixels)
            reg_preds = ltrb2xyxy(pixels_reflect, reshape(reg_head))        #(b, nums_pixels, 4)
            reg_preds = clipbbox(imgs, reg_preds)
            out_test = cls_prob, cls_class, reg_preds
        return loss_train if self.training else out_test


            
            
        

        
        
        

        


   
