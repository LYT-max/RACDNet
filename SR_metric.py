# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:09:19 2024

@author: Admin
"""

import os  
from glob import glob  
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim  
from PIL import Image  
import numpy as np  

def calculate_metrics(gt_folder, sr_folder):  
    gt_images = sorted(glob(os.path.join(gt_folder, '*.tif')))  
    sr_images = sorted(glob(os.path.join(sr_folder, '*.png')))  


    if len(gt_images) != len(sr_images):
        raise ValueError("The number of ground truth images and super-resolved images do not match.")
        
    total_psnr = 0  
    total_ssim = 0  
    count = 0  

    for gt_img_path, sr_img_path in zip(gt_images, sr_images):  

        gt_img = Image.open(gt_img_path).convert('L')  
        sr_img = Image.open(sr_img_path).convert('L')  

        if gt_img.size != sr_img.size:  
            raise ValueError(f"Images {gt_img_path} and {sr_img_path} have different sizes")  

        gt_img_np = np.array(gt_img, dtype=np.float32) / 255.0  
        sr_img_np = np.array(sr_img, dtype=np.float32) / 255.0  

        current_psnr = psnr(gt_img_np, sr_img_np, data_range=1.0)  
        current_ssim = ssim(gt_img_np, sr_img_np, data_range=1.0)  

        total_psnr += current_psnr  
        total_ssim += current_ssim  
        count += 1
              


    average_psnr = total_psnr / count  
    average_ssim = total_ssim / count  

    return average_psnr, average_ssim 
  
# 使用示例  
gt_folder = 'E:/high-level and low level task/SRCDNet-main/SRCDNet-main/Data/WHU_CD/test/time2'  
sr_folder = './results/WHU_CD/CDNet_ablation_1_2block/SR_results'  
average_psnr, average_ssim = calculate_metrics(gt_folder, sr_folder)  
print(f"Average PSNR: {average_psnr}")  
print(f"Average SSIM: {average_ssim}")