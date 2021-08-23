import torch
import torch.nn as nn

class Fcos_head(nn.Module):
    '''
        P3 -> 4 * (conv(3x3) + gn + relu) + conv(3x3) + (4 * (conv(3x3) + gn + relu) + conv(3x3))' [two side don't share weights]
                 => cls + cnt + reg
        P4 -> 4 * (conv(3x3) + gn + relu) + conv(3x3) + (4 * (conv(3x3) + gn + relu) + conv(3x3))' [two side don't share weights]
                 => cls + cnt + reg
        P5 -> 4 * (conv(3x3) + gn + relu) + conv(3x3) + (4 * (conv(3x3) + gn + relu) + conv(3x3))' [two side don't share weights]
                 => cls + cnt + reg
        P6 -> 4 * (conv(3x3) + gn + relu) + conv(3x3) + (4 * (conv(3x3) + gn + relu) + conv(3x3))' [two side don't share weights]
                 => cls + cnt + reg
        P7 -> 4 * (conv(3x3) + gn + relu) + conv(3x3) + (4 * (conv(3x3) + gn + relu) + conv(3x3))' [two side don't share weights]
                 => cls + cnt + reg
    '''
    def __init__(self, features, num_classes):
        super(Fcos_head, self).__init__()

        cls_moudle, reg_moudle = [], []
        for i in range(4):
            cls_moudle.append(nn.Conv2d(features, features, 3, padding = 1, bias = True))
            cls_moudle.append(nn.GroupNorm(32, features))
            cls_moudle.append(nn.ReLU(True))

            reg_moudle.append(nn.Conv2d(features, features, 3, padding = 1, bias = True))
            reg_moudle.append(nn.GroupNorm(32, features))
            reg_moudle.append(nn.ReLU(True))

        self.cls_conv_block = nn.Sequential(*cls_moudle)
        self.reg_conv_block = nn.Sequential(*reg_moudle)
    
        self.cls_conv = nn.Conv2d(features, num_classes, kernel_size = 3, padding = 1)
        self.reg_conv = nn.Conv2d(features, 4, kernel_size = 3, padding = 1)
        self.cnt_conv = nn.Conv2d(features, 1, kernel_size = 3, padding = 1)
        
        self.apply(self.init_conv_RandomNormal)

    def init_conv_RandomNormal(self, module, std = 0.01):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std = std)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        '''
            Args:
                x list[P3,P4,P5,P6,P7] from FPN
            Return:
                (self.cls_head, self.reg_head, self.cnt_head) tuple
                self.cls_head: 
                            list[tensor1,...]  len=5(layers feature_maps)
                            where                tensor1(Batch_size, nums_class, h, w)          
                self.reg_head: list[tensor1,...] tensor1(Batch_size, 4, h, w) with (l,t,r,b)
                self.cnt_head: list[tensor1,...] tensor1(Batch_size, 1, h, w)
                
            Attention:
                1) cnt_head'a conv_block shared weights with reg_head's conv_block
                2) here scale updated is abandoned in torch.exp(scale * feature_out)
        '''
        cls_head, cnt_head, reg_head = [], [], []
        for p in x:
            cls_head.append(self.cls_conv(self.cls_conv_block(p)))
            reg_head.append( torch.exp(self.reg_conv(self.reg_conv_block(p))) )
            cnt_head.append( torch.exp(self.cnt_conv(self.reg_conv_block(p))) )
        return cls_head, reg_head, cnt_head



        


    

    
