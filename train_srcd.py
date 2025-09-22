# coding=utf-8
import os
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import pandas as pd
from loss.losses import cross_entropy, WeightedCrossEntropy
from math import log10
from configures import parser
from loss.BCL import BCL
from loss.Gloss import GeneratorLoss
from data_utils import LoadDatasetFromFolder, calMetric_iou
from model.CDNet_ablation_1_2block_no_channel_attention import CDNet
from model.SRNet_edge import Generator
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
import numpy as np
import ever as er
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set seeds
def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
seed_torch(2021)

if __name__ == '__main__':
    mloss = 0

    # load data
    train_set = LoadDatasetFromFolder(args, args.hr1_train, args.lr2_train, args.hr2_train, args.lab_train)
    val_set = LoadDatasetFromFolder(args, args.hr1_val, args.lr2_val, args.hr2_val, args.lab_val)
    train_loader = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batchsize, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=args.num_workers, batch_size=args.val_batchsize, shuffle=True)

    # define model
    CDNet = CDNet().to(device, dtype=torch.float)
    netG = Generator(args.scale).to(device, dtype=torch.float)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        CDNet = torch.nn.DataParallel(CDNet, device_ids=range(torch.cuda.device_count()))
        netG = torch.nn.DataParallel(netG, device_ids=range(torch.cuda.device_count()))


    # set optimization
    optimizer = optim.Adam(CDNet.parameters(), lr= args.lr, betas=(0.9, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.001, betas=(0.9, 0.999))

    criterionCD = cross_entropy().to(device, dtype=torch.float)
    criterionG = GeneratorLoss(args.w_cd).to(device, dtype=torch.float)
    PLoss = nn.L1Loss(size_average=True).cuda()


    results = {'train_loss':[], 'train_CD':[], 'train_SR':[],'val_IoU':[]}

    # training
    for epoch in range(1, args.num_epochs + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'SR_loss':0, 'd_score':0,'g_score':0,'CD_loss':0, 'g_loss': 0 ,'d_loss': 0 }

        CDNet.train()
        netG.train()
        for hr_img1, lr_img2, hr_img2, label in train_bar:
            running_results['batch_sizes'] += args.batchsize

            hr_img1 = hr_img1.to(device, dtype=torch.float)
            lr_img2 = lr_img2.to(device, dtype=torch.float)
            hr_img2 = hr_img2.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.float)
            label = torch.argmax(label, 1).unsqueeze(1).float()
            scale2 = hr_img2.shape[3]/lr_img2.shape[3]
            scale1 = scale2 

            SR_img2 = netG(lr_img2, scale1, scale2)
            dist, dist1, dist2 = CDNet(hr_img1, SR_img2) # (img/0.5-1) aims to normalized the value to [-1, 1]

            ############################
            # Update CD network
            ###########################

            CD_loss = criterionCD(dist, label)+criterionCD(dist1, label)+criterionCD(dist2, label)
            CDNet.zero_grad()
            CD_loss.backward(retain_graph=True)
            optimizer.step()


            ############################

            ############################
            # Update G network
            ###########################
            netG.zero_grad()
            g_loss = PLoss(SR_img2, hr_img2)
            g_loss.backward()
            optimizerG.step()

            # loss for current batch before optimization
            running_results['CD_loss'] += CD_loss.item() * args.batchsize
            running_results['g_loss'] += g_loss.item() * args.batchsize

            train_bar.set_description(desc='[%d/%d] G: %.3f  CD_loss: %.3f' % (
                epoch, args.num_epochs, 
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['CD_loss'] / running_results['batch_sizes'],))

        # eval
        CDNet.eval()
        netG.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            inter, unin = 0,0
            valing_results = {'loss':0,'SR_loss': 0, 'CD_loss':0, 'batch_sizes': 0, 'IoU': 0, 'mse':0, 'psnr':0, 'ssim':0}
            metric_op = er.metric.PixelMetric(2, logdir=None, logger=None)

            for hr_img1, lr_img2, hr_img2, label in val_bar:
                valing_results['batch_sizes'] += args.val_batchsize

                hr_img1 = hr_img1.to(device, dtype=torch.float)
                lr_img2 = lr_img2.to(device, dtype=torch.float)
                hr_img2 = hr_img2.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.float)
                label = torch.argmax(label, 1).unsqueeze(1).float()
                scale2 = hr_img2.shape[3]/lr_img2.shape[3]
                scale1 = scale2 

                SR_img2 = netG(lr_img2, scale1, scale2)
                dist, _, _ = CDNet(hr_img1, SR_img2)

                # calculate IoU
                cd_map = torch.argmax(dist, 1).unsqueeze(1).float()
                gt_value = (label > 0).float()
                prob = (cd_map > 0).float()
                prob = prob.cpu().detach().numpy()
                

                gt_value = gt_value.cpu().detach().numpy()
                gt_value = np.squeeze(gt_value)
                result = np.squeeze(prob)
                
                metric_op.forward(gt_value, result)
                

                batch_mse = ((SR_img2 - hr_img2) ** 2).data.mean()
                valing_results['mse'] += batch_mse * args.val_batchsize
                valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))

                val_bar.set_description(
                    desc='PSNR: %.4f' % (valing_results['psnr']))
                
                
            re = metric_op.summary_all()

        # save model parameters
        val_loss = valing_results['IoU']


        torch.save(CDNet.state_dict(),  args.model_dir+'netCD_epoch_%d.pth' % epoch)
        torch.save(netG.state_dict(), args.sr_dir + 'netG_epoch_%d.pth' % epoch)

        results['train_SR'].append(running_results['SR_loss'] / running_results['batch_sizes'])
        results['train_CD'].append(running_results['CD_loss'] / running_results['batch_sizes'])
        results['val_IoU'].append(valing_results['IoU'])

        if epoch % 10 == 0 and epoch != 0:
            data_frame = pd.DataFrame(
                data={'train_CD': results['train_CD'],
                      'val_IoU': results['val_IoU']},
                index=range(1, epoch + 1))
            data_frame.to_csv(args.sta_dir, index_label='Epoch')
