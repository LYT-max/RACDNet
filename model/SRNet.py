import math
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F_torch

import torch.nn.functional as F

def grid_sample(x, offset, scale, scale2):
    # generate grids
    b, _, h, w = x.size()
    grid = np.meshgrid(range(round(scale2*w)), range(round(scale*h)))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid = torch.Tensor(grid).to(x.device)

    # project into LR space 进行归一化处理，但考虑到像素中心。
    grid[:, :, 0] = (grid[:, :, 0] + 0.5) / scale2 - 0.5
    grid[:, :, 1] = (grid[:, :, 1] + 0.5) / scale - 0.5

    # normalize to [-1, 1]
    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) -1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) -1
    
    grid = grid.permute(2, 0, 1).unsqueeze(0)
    grid = grid.expand([b, -1, -1, -1])


    # add offsets
    offset_0 = torch.unsqueeze(offset[:, 0, :, :] * 2 / (w - 1), dim=1)
    offset_1 = torch.unsqueeze(offset[:, 1, :, :] * 2 / (h - 1), dim=1)
    grid = grid + torch.cat((offset_0, offset_1),1)
    grid = grid.permute(0, 2, 3, 1)
    # sampling
    output = F.grid_sample(x, grid, padding_mode='zeros')

    return output


# Sobel filter for image gradient computation


class SA_upsample(nn.Module):
    def __init__(self, channels, num_experts=4, bias=False):
        super(SA_upsample, self).__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels

        # experts
        weight_compress = []
        for i in range(num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(channels // 8, channels, 1, 1)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        weight_expand = []
        for i in range(num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(channels, channels // 8, 1, 1)))
            nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))

        # two FC layers
        self.body = nn.Sequential(
            nn.Conv2d(4, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        # routing head
        self.routing = nn.Sequential(
            nn.Conv2d(64, num_experts, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        # offset head
        self.offset = nn.Conv2d(64, 2, 1, 1, 0, bias=True)
        self.conv = nn.Conv2d(64*3, 64, 1, 1, 0, bias=True)
        
    def compute_gradients(self, x):
        # Sobel filter (with 1 input channel and 1 output channel)
        sobel_filter_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_filter_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Extend the filters to match the number of input channels
        sobel_filter_x = sobel_filter_x.repeat(x.size(1), 1, 1, 1).to(x.device)
        sobel_filter_y = sobel_filter_y.repeat(x.size(1), 1, 1, 1).to(x.device)
        
        # Compute gradients using F.conv2d
        grad_x = F.conv2d(x, sobel_filter_x, padding=1, groups=x.size(1))  # Apply Sobel filter to each input channel separately
        grad_y = F.conv2d(x, sobel_filter_y, padding=1, groups=x.size(1))  # Apply Sobel filter to each input channel separately
    
        return grad_x, grad_y



    def forward(self, x, scale, scale2):
        b, c, h, w = x.size()

        # Compute gradients of the input image
        grad_x, grad_y = self.compute_gradients(x)
        
        # (1) coordinates in LR space
        ## coordinates in HR space
        coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(x.device),
                   torch.arange(0, round(w * scale2), 1).unsqueeze(0).float().to(x.device)]
        
        
        ## coordinates in LR space
        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5  #torch.Size([384, 1])  Relative distance
        coor_h = coor_h.permute(1, 0)
        coor_w = ((coor_hr[1] + 0.5) / scale2) - (torch.floor((coor_hr[1] + 0.5) / scale2 + 1e-3)) - 0.5 #torch.Size([1, 384])  Relative distance
        

        input = torch.cat((
            torch.ones_like(coor_h).expand([-1, round(scale2 * w)]).unsqueeze(0) / scale2,
            torch.ones_like(coor_h).expand([-1, round(scale2 * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, round(scale2 * w)]).unsqueeze(0),
            coor_w.expand([round(scale * h), -1]).unsqueeze(0)
        ), 0).unsqueeze(0)  #torch.Size([1, 4, 384, 384])
        

        # (2) predict filters and offsets
        embedding = self.body(input) #torch.Size([1, 64, 384, 384])
        
        # offsets
        offset = self.offset(embedding) #torch.Size([1, 2, 384, 384])
        
        # filters
        routing_weights = self.routing(embedding) #torch.Size([1, 4, 384, 384])
        

        routing_weights = routing_weights.view(self.num_experts, round(scale * h) * round(scale2 * w)).transpose(0, 1)  # (h*w) * n torch.Size([147456, 4])     
        
        weight_compress = self.weight_compress.view(self.num_experts, -1)  #torch.Size([4, 512])          #self.weight_compress  torch.Size([4, 8, 64, 1, 1])
        
        weight_compress = torch.matmul(routing_weights, weight_compress)  #torch.Size([147456, 512])
        
        weight_compress = weight_compress.view(1, round(scale * h), round(scale2 * w), self.channels // 8, self.channels)  #torch.Size([1, 384, 384, 8, 64])

        weight_expand = self.weight_expand.view(self.num_experts, -1)  #torch.Size([4, 512])   #self.weight_expand  torch.Size([4, 8, 64, 1, 1])
               
        weight_expand = torch.matmul(routing_weights, weight_expand)  #torch.Size([147456, 512])  
               
        weight_expand = weight_expand.view(1, round(scale * h), round(scale2 * w), self.channels, self.channels // 8)  #torch.Size([1, 384, 384, 64, 8])
        

        # (3) Incorporate gradients into the features before sampling
        # Adding gradients as additional channels to the input
        grad_combined = torch.cat([grad_x, grad_y], dim=1)  # (b, 2, h, w)  torch.Size([1, 128, 256, 256])
      
        # Concatenate gradients with the original input for better feature extraction
        x_combined = torch.cat([x, grad_combined], dim=1)  # (b, c+2, h, w)   torch.Size([1, 128, 256, 256])
        
        # (4) grid sample & spatially varying filtering
        # Perform grid sampling
        fea0 = grid_sample(self.conv(x_combined), offset, scale, scale2)  # b * h * w * c * 1  torch.Size([1, 64, 384, 384])
        
        # Apply spatially varying filtering
# Assuming fea0 has shape [b, h, w, c]

        fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)  # b * h * w * c * 1   torch.Size([1, 384, 384, 64, 1])
        out = torch.matmul(weight_compress.expand([b, -1, -1, -1, -1]), fea)  #torch.Size([1, 384, 384, 8, 1])    
        out = torch.matmul(weight_expand.expand([b, -1, -1, -1, -1]), out).squeeze(-1)  #torch.Size([1, 384, 384, 64])

        return out.permute(0, 3, 1, 2) + fea0 #torch.Size([1, 64, 384, 384])








class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.adapt_up = SA_upsample(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)
        self.conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x, scale1, scale2):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block8 = self.block7(block6)
        block8 = self.conv(self.adapt_up(block8, scale1, scale2))

        return block8
    





class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

model = SA_upsample(64)
a = torch.randn(1, 64, 256, 256)
out = model(a, 1.5, 1.5)
print(out.shape)



