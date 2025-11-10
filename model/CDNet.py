import torch
from torch import nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from torch import Tensor
from torch import Tensor, reshape, stack
#from .backbone import build_backbone
from .modules import TransformerDecoder, Transformer
from einops import rearrange
from torch.nn import Upsample
from .backbone import build_backbone

import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import numpy as np  
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch.nn.init as init


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, d, relu_slope=0.1):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.relu_1(self.conv_1(x))
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, d = 1, init='xavier', gc=8, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc, d)
        self.conv2 = UNetConvBlock(gc, gc, d)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3


    

  
# 定义一个简单的卷积层  
# class Freprocess(nn.Module):
#     def __init__(self, channels):
#         super(Freprocess, self).__init__()
#         self.pre1 = nn.Conv2d(channels,channels,1,1,0)
#         self.pre2 = nn.Conv2d(channels,channels,1,1,0)
#         self.amp_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
#                                       nn.Conv2d(channels,channels,1,1,0))
#         self.pha_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
#                                       nn.Conv2d(channels,channels,1,1,0))
#         self.post = nn.Conv2d(channels,channels,1,1,0)

#     def forward(self, img1, img2):

#         _, _, H, W = img1.shape
#         img1 = torch.fft.rfft2(self.pre1(img1)+1e-8, norm='backward').cpu()
#         img2 = torch.fft.rfft2(self.pre2(img2)+1e-8, norm='backward').cpu()
#         img1_amp = torch.abs(img1).cuda()
#         img1_pha = torch.angle(img1).cuda()
#         img2_amp = torch.abs(img2).cuda()
#         img2_pha = torch.angle(img2).cuda()
#         amp_fuse = self.amp_fuse(img1_amp-img2_amp)
#         pha_fuse = self.pha_fuse(img1_pha-img2_pha)

#         real = amp_fuse * torch.cos(pha_fuse)+1e-8
#         imag = amp_fuse * torch.sin(pha_fuse)+1e-8
#         out = torch.complex(real, imag)+1e-8
#         out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))

#         return self.post(out)

class Freprocess(nn.Module):
    def __init__(self, channels):
        super(Freprocess, self).__init__()
        self.pre1 = nn.Conv2d(channels,channels,1,1,0)
        self.pre2 = nn.Conv2d(channels,channels,1,1,0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, img1, img2):

        _, _, H, W = img1.shape
        img1 = torch.fft.rfft2(self.pre1(img1-img2)+1e-8, norm='backward').cpu()
   
        img1_amp = torch.abs(img1).cuda()
        img1_pha = torch.angle(img1).cuda()
  
        amp_fuse = self.amp_fuse(img1_amp)
        pha_fuse = self.pha_fuse(img1_pha)

        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))

        return self.post(out)
    
class SpaFre(nn.Module):
    def __init__(self, channels):
        super(SpaFre, self).__init__()
        self.conv1 = nn.Conv2d(channels,channels,3,1,1)
        self.conv2 = nn.Conv2d(channels,channels,3,1,1)
        self.conv3 = nn.Conv2d(channels*2,channels,1,1,0)
        self.spa_process = DenseBlock(channels, channels)
        self.fre_process = Freprocess(channels)
        self.spa_att = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.cha_att = nn.Sequential(nn.Conv2d(channels * 2, channels // 2, kernel_size=1, padding=0, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels * 2, kernel_size=1, padding=0, bias=True),
                                     nn.Sigmoid())
        self.post = nn.Conv2d(channels * 2, channels, 3, 1, 1)

    def forward(self, img1, img2):  #, i
        img1 = self.conv1(img1)
        img2 = self.conv2(img2)
        spafuse = self.spa_process(img1-img2)
        frefuse = self.fre_process(img1,img2)
        spa_map = self.spa_att(self.conv3(torch.cat([spafuse, frefuse],1)))
        cat_f = torch.cat([frefuse*spa_map,spafuse*spa_map],1)
        cha_res =  self.post(self.cha_att(self.avgpool(cat_f))*cat_f)
        out = cha_res+img1+img2

        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)




    




class Classifier(nn.Module):
    def __init__(self, in_chan=32, n_class=2):
        super(Classifier, self).__init__()
        self.head = nn.Sequential(
                            nn.Conv2d(in_chan, in_chan, kernel_size=3, padding=1, stride=1, bias=False),
                            nn.BatchNorm2d(in_chan),
                            nn.ReLU(),
                            nn.Conv2d(in_chan, n_class, kernel_size=3, padding=1, stride=1))
    def forward(self, x):
        x = self.head(x)
        return x
    
class Classifier1(nn.Module):
    def __init__(self, in_chan=32, n_class=2):
        super(Classifier1, self).__init__()
        self.head = nn.Sequential(
                            nn.Conv2d(in_chan*2, in_chan, kernel_size=3, padding=1, stride=1, bias=False),
                            nn.BatchNorm2d(in_chan),
                            nn.ReLU(),
                            nn.Conv2d(in_chan, n_class, kernel_size=3, padding=1, stride=1))
    def forward(self, x):
        x = self.head(x)
        return x
    
class Conv_stride(nn.Module):
    def __init__(self,in_channels=32):
        super(Conv_stride, self).__init__()
        self.conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        self.batchNorm = nn.BatchNorm2d(32, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.relu(x)
        return x

class Upsample(nn.Module):
    def __init__(self,scale_factor=2):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchNorm = nn.BatchNorm2d(64, momentum=0.1)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="bicubic", align_corners=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.upsample(x)
        return x

class CDNet(nn.Module):
    def __init__(self,  backbone='resnet50', output_stride=16, img_size = 256, img_chan=3, chan_num = 32, n_class =2):
        super(CDNet, self).__init__()
        BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm, img_chan)
        self.SEIB1 = SpaFre(32)
        self.SEIB2 = SpaFre(32)
        self.SEIB3 = SpaFre(32)
        self.SEIB4 = SpaFre(32)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.classifier = Classifier1(n_class = n_class)
        self.classifier1 = Classifier(n_class = n_class)
        self.classifier2 = Classifier(n_class = n_class)
        self.classifier3 = Classifier(n_class = n_class)

        self.sam = SpatialAttention()


    def forward(self, img1, img2):
        # CNN backbone, feature extractor
        _, out1_s8, out1_s4 = self.backbone(img1)


        _, out2_s8, out2_s4 = self.backbone(img2)
        

        out2 = self.SEIB2(out1_s8, out2_s8)
        out3 = self.SEIB3(out1_s4, out2_s4)
        


        x8 = F.interpolate(out2, size=img1.shape[2:], mode='bicubic', align_corners=True)
        x = F.interpolate(out3, size=img1.shape[2:], mode='bicubic', align_corners=True)

        out = self.classifier(torch.cat([x, x8], dim=1))
        x = self.classifier3(x)
        x8 = self.classifier2(x8)


        return out, x, x8

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
