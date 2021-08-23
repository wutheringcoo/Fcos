from model.backbone import resnet50

import torch.nn as nn
import torch.nn.functional  as F
class Fpn(nn.Module):
    '''
        P5,P4,P5 from C5''(C5),C4'',C3'':
            Serial:
                1) C5' : C5(1x1 conv)
                        return (B,features,H/32,W/32)
                2) C4' : C4(1x1 conv) + C5'(upsample) [add]
                        return (B,features,H/16,W/16) + (B,features,H/16,W/16)
                3) C3' : C3(1x1 conv) + C4'(upsample) [add]
                        return (B,features,H/8,W/8) + (B,features,H/8,W/8)
            Parallel:
                4):
                    P5 = C5'' : C5'(3x3 conv stride=2) return (B,features,H/64,W/64)
                    P4 = C4'' : C4'(3x3 conv stride=2) return (B,features,H/32,W/32)
                    P3 = C3'' : C3'(3x3 conv stride=2) return (B,features,H/16,W/16)
                    if use_p5:
                        4) P5 = C5 return (B,2048,H/32,W/32)
        
        P6, P7 from P5:
                5) P6  P5(3x3 conv stride=2) return (B,features,H/128,W/128)
                   if use_p5:
                       P6 = C5(3x3 conv stride=2)  return (B,features,H/64,W/64)
                6) P7  from P6 (relu => 3x3 conv stride=2) return (B,features,H/256,W/256)
    '''
    def __init__(self, features):
        super(Fpn, self).__init__()
        self.pconv_5 = nn.Conv2d(2048, features, kernel_size = 1)
        self.pconv_4 = nn.Conv2d(1024, features, kernel_size = 1)
        self.pconv_3 = nn.Conv2d(512, features, kernel_size = 1)
        self.conv_3 = nn.Conv2d(features, features, kernel_size = 3, padding = 1)
        self.conv_4 = nn.Conv2d(features, features, kernel_size = 3, padding = 1)
        self.conv_5 = nn.Conv2d(features, features, kernel_size = 3, padding = 1)
        self.conv_6 = nn.Conv2d(features, features, kernel_size = 3, padding = 1, stride=2)
        self.conv_7 = nn.Conv2d(features, features, kernel_size = 3, padding = 1, stride=2)
        self.apply(self.init_conv_kaiming)

    def init_conv_kaiming(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        '''
            Args:
                x's generated from backbone
        '''
        
        C3, C4, C5 = x
        P5 = self.pconv_5(C5)
        # return [C5, P5]
        P4 = self.pconv_4(C4) + F.interpolate(P5, size = (C4.shape[2], C4.shape[3]), mode = 'nearest' )
        P3 = self.pconv_3(C3) + F.interpolate(P4, size = (C3.shape[2], C3.shape[3]), mode = 'nearest' )
        P5 = self.conv_5(P5)
        P4 = self.conv_4(P4)
        P3 = self.conv_3(P3)

        P6 = self.conv_6(P5)
        P7 = self.conv_7(F.relu(P6))
        return [P3,P4,P5,P6,P7]

if __name__ == "__main__":
    import torch
    from model.backbone import resnet50
    img = torch.rand(32, 3, 800, 1024)
    weights_path = '/data-nbd/wuxc/Resource/weights/Restnets/resnet50-19c8e357.pth'
    result = resnet50(pretrained = True, weights = weights_path, num_classes= 5)(img)
    '''
        C3 C4 C5:
        [([32, 512, 100, 128]) ,([32, 1024, 50, 64]), ([32, 2048, 25, 32])]
    '''
    # print(result[0].shape, result[1].shape, result[2].shape) 
    P3,P4,P5,P6,P7 = FPN(features = 4)(result)
    '''
        [
         ([32, 4, 100, 128]),
         ([32, 4, 50, 64]),
         ([32, 4, 25, 32]) , 
         ([32, 4, 13, 16]),
         ([32, 4, 7, 8]),
         ]
    '''
    print(P3.shape, P4.shape, P5.shape, P6.shape, P7.shape)
    
    
    




