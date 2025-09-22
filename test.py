# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 10:10:22 2024

@author: Admin
"""

# coding=utf-8
# coding=utf-8
import os
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import pandas as pd
from math import log10
from configures import parser
from thop import profile  # 导入thop库用于计算FLOPs
from loss.Gloss import GeneratorLoss
from data_utils import LoadDatasetFromFolder, calMetric_iou
from model.CDNet_ablation_1_2block import CDNet
from model.SRNet_edge import Generator
import torch.nn as nn
import cv2
import torchvision.utils as vutils 
import ever as er
from PIL import Image, ImageDraw
args = parser.parse_args()



def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1e6

# 添加计算FLOPs的函数
def count_flops(model, input_tensor):
    flops, params = profile(model, inputs=(input_tensor,))
    return flops, params



parser.add_argument('--save_cd', default='./results', type=str)



args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(args.save_cd):
    os.mkdir(args.save_cd)
    
COLOR_MAP = {'0': (0, 0, 0),  # black is TN  
             '1': (0, 255, 0),  # green is FP 误检  
             '2': (255, 0, 0),  # red is FN 漏检  
             '3': (255, 255, 255)}  # white is TP  
  
def generate_colored_image(gt_value, result, color_map):  
    # 假设gt_value和result都是二维numpy数组，形状相同  
    height, width = gt_value.shape  
    colored_img = Image.new('RGB', (width, height))  
    draw = ImageDraw.Draw(colored_img)  
    for i in range(height):  
        for j in range(width):  
            # 比较gt_value和result的值  
            if gt_value[i, j] == 0 and result[i, j] == 0:  # TN  
                color = color_map['0']  
            elif gt_value[i, j] == 0 and result[i, j] > 0:  # FP  
                color = color_map['1']  
            elif gt_value[i, j] > 0 and result[i, j] == 0:  # FN  
                color = color_map['2']  
            elif gt_value[i, j] > 0 and result[i, j] > 0:  # TP  
                color = color_map['3']  
            draw.point((j, i), fill=color)  
  
    return colored_img
    


if __name__ == '__main__':
    mloss = 0
    metric_op = er.metric.PixelMetric(2, logdir=None, logger=None)

    # load data

    test_set = LoadDatasetFromFolder(args, args.hr1_test, args.lr2_test, args.hr2_test, args.lab_test)
    test_loader = DataLoader(dataset=test_set, num_workers=args.num_workers, batch_size=args.test_batchsize, shuffle=True)

    # define model
    sample_hr1, sample_lr2, _, _, _ = next(iter(test_loader))
    sample_hr1 = sample_hr1.to(device, dtype=torch.float)
    sample_lr2 = sample_lr2.to(device, dtype=torch.float)
    
    # define model
    CDNet = CDNet().to(device, dtype=torch.float)
    netG = Generator(args.scale).to(device, dtype=torch.float)
    
    CDNet.load_state_dict(torch.load(args.test_model_dir))
    netG.load_state_dict(torch.load(args.test_sr_dir))

    # 计算参数量
    cdnet_params = count_parameters(CDNet)
    netG_params = count_parameters(netG)
    total_params = cdnet_params + netG_params
    sample_hr1 = torch.randn(1, 3, 256, 256).cuda()
    sample_lr2 = torch.randn(1, 3, 64, 64).cuda()
    
    # 计算FLOPs
    # 为CDNet准备输入
    scale2 = sample_hr1.shape[3]/sample_lr2.shape[3]
    scale1 = scale2
    SR_img2 = netG(sample_lr2, scale1, scale2)
    
    # 计算FLOPs (需要根据模型输入调整)
    cdnet_flops, _ = profile(CDNet, inputs=(sample_hr1, SR_img2))


    netG_flops, _ = profile(netG, inputs=(sample_lr2, scale1, scale2))
    total_flops = cdnet_flops + netG_flops
    
    # 打印参数量和FLOPs
    print(f"CDNet Parameters: {cdnet_params:,}")
    print(f"Generator Parameters: {netG_params:,}")
    print(f"Total Parameters: {total_params:,}")
    print(f"CDNet FLOPs: {cdnet_flops/1e9:.2f} GFLOPs")
    print(f"Generator FLOPs: {netG_flops/1e9:.2f} GFLOPs")
    print(f"Total FLOPs: {total_flops/1e9:.2f} GFLOPs")


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        CDNet = torch.nn.DataParallel(CDNet, device_ids=range(torch.cuda.device_count()))
        netG = torch.nn.DataParallel(netG, device_ids=range(torch.cuda.device_count()))




    results = {'train_loss':[], 'train_CD':[], 'train_SR':[],'test_IoU':[]}



        # etest
    
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        inter, unin = 0,0
        testing_results = {'loss':0,'SR_loss': 0, 'CD_loss':0, 'batch_sizes': 0, 'IoU': 0, 'mse':0, 'psnr':0}

        for hr_img1, lr_img2, hr_img2, label, name in test_bar:
            testing_results['batch_sizes'] += args.test_batchsize
            filename_with_path = name[0]
            base_name = os.path.basename(filename_with_path)  # 假设name是完整的文件路径  
            filename_without_ext, file_extension = os.path.splitext(base_name)

            hr_img1 = hr_img1.to(device, dtype=torch.float)
            lr_img2 = lr_img2.to(device, dtype=torch.float)
            hr_img2 = hr_img2.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.float)
            label = torch.argmax(label, 1).unsqueeze(1).float()
            scale2 = hr_img2.shape[3]/lr_img2.shape[3]
            scale1 = scale2 

            SR_img2 = netG(lr_img2, scale1, scale2)
            dist, _, _ = CDNet(hr_img1, SR_img2)
            SR_img2 = SR_img2.clamp(0, 1)  # 如果SR_img2的值不在[0, 1]范围内，这一步是必要的  
      
    # 假设你想将每个生成的图像保存为PNG格式  
            save_path = os.path.join('./results/WHU_CD/end_to_end/SR_results', f'{filename_without_ext}.png')  
            vutils.save_image(SR_img2, save_path, normalize=True, nrow=1, padding=0)  # normalize=True是因为SR_img2的值在[0, 1]
            cd_map = torch.argmax(dist, 1).unsqueeze(1).float()
            gt_value = (label > 0).float()
            prob = (cd_map > 0).float()
            prob = prob.cpu().detach().numpy()
            

            gt_value = gt_value.cpu().detach().numpy()
            gt_value = np.squeeze(gt_value)
            result = np.squeeze(prob)
            
            # SR_img2 = SR_img2.squeeze(0)
            # print(SR_img2)
 

            metric_op.forward(gt_value, result)
            
            colored_img = generate_colored_image(gt_value, result, COLOR_MAP)
            result_img = Image.fromarray((result * 255).astype(np.uint8))  # 转换为0-255的整数  
            # SR_img2 = Image.fromarray((SR_img2 * 255))  # 转换为0-255的整数  
            save_path = os.path.join('./results/WHU_CD/end_to_end/pre', f'{filename_without_ext}.png')  # 根据需要修改文件名  
            result_img.save(save_path)
            
            save_path = os.path.join('./results/WHU_CD/end_to_end/save', f'{filename_without_ext}.png')  
            colored_img.save(save_path) 
            
            # save_path = os.path.join('./results/SR_results', f'{filename_without_ext}.png')  
            # SR_img2.save(save_path) 
            
            batch_mse = ((SR_img2 - hr_img2) ** 2).data.mean()
            testing_results['mse'] += batch_mse * args.test_batchsize
            testing_results['psnr'] = 10 * log10(1 / (testing_results['mse'] / testing_results['batch_sizes']))
            
            testing_results['IoU'] = (inter * 1.0 /( unin+0.000000001))
            test_bar.set_description(
                desc='IoU: %.4f   PSNR: %.4f' % (testing_results['IoU'],testing_results['psnr']))

            
        re = metric_op.summary_all()



            



