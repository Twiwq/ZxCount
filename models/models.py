import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange
from math import sqrt
from trainers.vgg_decoder_high import VGGDecoder_high

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, bn=False, relu=True,groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias,groups=groups)
        # self.bn = BatchRenorm2d(out_channels) if bn else None
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            y = self.bn(y)
        if self.relu is not None:
            y = self.relu(y)
        return y




        
def upsample(x, scale_factor=2, mode='bilinear'):
    if mode == 'nearest':
        return F.interpolate(x, scale_factor=scale_factor, mode=mode)
    else:
        return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=False)


class DGModel_base(nn.Module):
    def __init__(self, pretrained=True, den_dropout=0.5):
        super().__init__()

        self.den_dropout = den_dropout

        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None)
        self.enc1 = nn.Sequential(*list(vgg.features.children())[:23])
        self.enc2 = nn.Sequential(*list(vgg.features.children())[23:33])
        self.enc3 = nn.Sequential(*list(vgg.features.children())[33:43])

        self.dec3 = nn.Sequential(
            ConvBlock(512, 1024, bn=True),
            ConvBlock(1024, 512, bn=True)
        )
    

        self.dec2 = nn.Sequential(
            ConvBlock(1024, 512, bn=True),
            ConvBlock(512, 256, bn=True)
        )

        self.dec1 = nn.Sequential(
            ConvBlock(512, 256, bn=True),
            ConvBlock(256, 128, bn=True)
        )
       
       

        self.den_dec = nn.Sequential(
            ConvBlock(512 + 256 + 128, 256, kernel_size=1, padding=0, bn=True),
        )

        self.den_head = nn.Sequential(
            ConvBlock(256, 1, kernel_size=1, padding=0)
        )
        
        self.err_head = nn.Sequential(
            ConvBlock(256, 256, kernel_size=1, padding=0, bn=False, relu=False),
            nn.Sigmoid()
        )
        

        self.conv_pool2 = nn.Sequential(
            ConvBlock(256, 256,kernel_size=2, padding=0,stride=2,groups=256)
        )   
        
        self.conv_enh = nn.Sequential(
            ConvBlock(256, 256,kernel_size=1, padding=0),
        )
        
     
        self.memory_list_global = nn.Parameter(torch.randn((24, 256)))
        self.memory_list_mask = nn.Parameter(torch.randn((24, 256)))
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        

        
    def forward_fe(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        x = self.dec3(x3)
        y3 = x
        x = upsample(x, scale_factor=2)
        x = torch.cat([x, x2], dim=1)

        x = self.dec2(x)
        y2 = x
        x = upsample(x, scale_factor=2)
        x = torch.cat([x, x1], dim=1)

        x = self.dec1(x)
        y1 = x

        y2 = upsample(y2, scale_factor=2)
        y3 = upsample(y3, scale_factor=4)

        y_cat = torch.cat([y1, y2, y3], dim=1)

        return y_cat, x3

    def forward(self, x):
        y_cat, _ = self.forward_fe(x)

        y_den = self.den_dec(y_cat)
        d = self.den_head(y_den)
        d = upsample(d, scale_factor=4)

        return d

class DGModel_cls(DGModel_base):
    def __init__(self, pretrained=True, den_dropout=0.5, cls_dropout=0.3, cls_thrs=0.5):
        super().__init__(pretrained, den_dropout)

        self.cls_dropout = cls_dropout
        self.cls_thrs = cls_thrs


    def transform_cls_map_gt(self, c_gt):
        return upsample(c_gt, scale_factor=4, mode='nearest')

    def transform_cls_map_pred(self, c):
        c_new = c.clone().detach()
        c_new[c < self.cls_thrs] = 0
        c_new[c >= self.cls_thrs] = 1
        c_resized = upsample(c_new, scale_factor=4, mode='nearest')

        return c_resized

    def transform_cls_map(self, c, c_gt=None):
        if c_gt is not None:
            return self.transform_cls_map_gt(c_gt)
        else:
            return self.transform_cls_map_pred(c)

    def forward(self, x, c_gt=None):
        y_cat, x3 = self.forward_fe(x)

        y_den = self.den_dec(y_cat)

        c = self.cls_head(x3)
        c_resized = self.transform_cls_map(c, c_gt)
        d = self.den_head(y_den)
        dc = d * c_resized
        dc = upsample(dc, scale_factor=4)

        return dc, c


class DGModel_memcls(DGModel_base):
    def __init__(self, pretrained=True,  den_dropout=0.5, cls_dropout=0.3, cls_thrs=0.5):
        super().__init__(pretrained,  den_dropout)

        self.cls_dropout = cls_dropout
        self.cls_thrs = cls_thrs

        self.mask_head = nn.Sequential(
            ConvBlock(256, 256, kernel_size=1, padding=0),
            ConvBlock(256, 25, kernel_size=1, padding=0, relu=False),
        )
        
        self.cls_head = nn.Sequential(
            ConvBlock(512, 256, bn=True),
            nn.Dropout2d(p=0.3),
            ConvBlock(256, 1, kernel_size=1, padding=0, relu=False),
            nn.Sigmoid()
        )


    

    def transform_cls_map_gt(self, c_gt):
        return upsample(c_gt, scale_factor=4, mode='nearest')
    
    def transform_cls_map_pred(self, c):
        c_new = c.clone().detach()
        c_new[c<self.cls_thrs] = 0
        c_new[c>=self.cls_thrs] = 1
        c_resized = upsample(c_new, scale_factor=4, mode='nearest')

        return c_resized

    
    def transform_cls_mask_pred(self, c):
        c_new = c.clone().detach()
        c_new[c<self.cls_thrs] = 0
        c_new[c>=self.cls_thrs] = 1
        c_resized = upsample(c_new, scale_factor=1, mode='nearest')

        return c_resized

    def transform_cls_map(self, c, c_gt=None):
        if c_gt is not None:
            return self.transform_cls_map_gt(c_gt)
        else:
            return self.transform_cls_map_pred(c)

    def forward(self, x, c_gt=None):

        y_cat, x3 = self.forward_fe(x)
        y_den = self.den_dec(y_cat)

       

        e_mask_pred = self.err_head(y_den)
        mask_bin = self.transform_cls_mask_pred(e_mask_pred)

        y_den_mask=y_den*mask_bin
        
        den1=self.conv_enh(y_den_mask)

        c = self.cls_head(x3)
        c_resized = self.transform_cls_map_pred(c)
        


        d = self.den_head(den1)*c_resized
        dc =  upsample(d*c_resized, scale_factor=4)
        
        return dc


class DGModel_final(DGModel_memcls):
    def __init__(self, pretrained=True,  cls_thrs=0.5, den_dropout=0.5,
                 cls_dropout=0.3, has_err_loss=False):
        super().__init__(pretrained,  den_dropout, cls_dropout, cls_thrs)

       

        self.den_dec = nn.Sequential(
            ConvBlock(512 + 256 + 128, 256, kernel_size=1, padding=0, bn=True)
        )

    def jsd(self, logits1, logits2):
        p1 = F.softmax(logits1, dim=1)
        p2 = F.softmax(logits2, dim=1)
        jsd = F.mse_loss(p1, p2)
        return jsd

    def IN(self, x1, x2, err=0.5):
        y_in1 = F.instance_norm(x1, eps=1e-5)
        y_in2 = F.instance_norm(x2, eps=1e-5)

        e_y = torch.abs(y_in1 - y_in2)
        e_mask = (e_y < err).clone().detach()
        e_mask_bg = (e_y >= err).clone().detach()
        
        x_masked1 = F.dropout2d(x1 * e_mask,0)
        x_masked2 = F.dropout2d(x2 * e_mask,0)
        x_masked_bg1 = F.dropout2d(x1 * e_mask_bg,0)
        x_masked_bg2 = F.dropout2d(x2 * e_mask_bg,0)
        
        return x_masked1, x_masked2, x_masked_bg1,x_masked_bg2,e_mask

    def forward_train(self, img1, img2, c_gt=None):

        y_cat1, x3_1 = self.forward_fe(img1)
        y_cat2, x3_2 = self.forward_fe(img2)
        y_den1 = self.den_dec(y_cat1)
        y_den2 = self.den_dec(y_cat2)  #80*80



        glo_down1=self.conv_pool2(y_den1)  #40*40
        glo_down2=self.conv_pool2(y_den2)
        _,_,_,_,e_mask_down=self.IN(glo_down1,glo_down2)
        e_mask_1=upsample(e_mask_down.float(),scale_factor=2,mode='nearest')

        _,_,_,_,e_mask_2=self.IN(y_den1,y_den2)

        e_mask = (e_mask_1 + e_mask_2)/2

        e_mask_pred1 = self.err_head(y_den1)
        e_mask_pred2 = self.err_head(y_den2)
        l_err = F.binary_cross_entropy(e_mask_pred1, e_mask.to(e_mask_pred1.dtype).detach()) + \
                F.binary_cross_entropy(e_mask_pred2, e_mask.to(e_mask_pred2.dtype).detach())


        y_den_mask1=y_den1*e_mask
        y_den_mask2=y_den2*e_mask
        y_bg1=y_den1*(1-e_mask)
        y_bg2=y_den2*(1-e_mask)



        den1=self.conv_enh(y_den_mask1)
        den2=self.conv_enh(y_den_mask2)
        y_bg1=self.conv_enh(y_bg1)
        y_bg2=self.conv_enh(y_bg2)

        fg_down1=self.conv_pool2(den1)
        fg_down2=self.conv_pool2(den2)
        bg_down1=self.conv_pool2(y_bg1) #40*40
        bg_down2=self.conv_pool2(y_bg2)


        cls_score_fg1 = self.mask_head(fg_down1)  #25 40 40
        cls_score_fg2 = self.mask_head(fg_down2)

        cls_score_max_fg1 = cls_score_fg1.max(dim=1, keepdim=True)[0]
        cls_score_max_fg2 = cls_score_fg2.max(dim=1, keepdim=True)[0]
        cls_score_fg1 = cls_score_fg1 - cls_score_max_fg1
        cls_score_fg2 = cls_score_fg2 - cls_score_max_fg2


        c1 = self.cls_head(x3_1)
        c2 = self.cls_head(x3_2)
        c_resized1 = self.transform_cls_map_pred(c1)
        c_resized2 = self.transform_cls_map_pred(c2)
        c_err = torch.abs(c_resized1 - c_resized2)
        c_resized_gt = self.transform_cls_map_gt(c_gt)
        c_resized = torch.clamp(c_resized_gt + c_err, 0, 1)


        d1 = self.den_head(den1) #80*80
        d2 = self.den_head(den2)
        dc1 = upsample(d1*c_resized,4)  #320*320
        dc2 = upsample(d2*c_resized,4)



        fg_dp1 = self.den_head(fg_down1) #80*80
        fg_dp2 = self.den_head(fg_down2)
        

        return (dc1, dc2),(glo_down1,glo_down2,fg_down1,fg_down2,bg_down1,bg_down2),l_err,(c1,c2), (cls_score_fg1,cls_score_fg2),(fg_dp1,fg_dp2)

