import argparse

#training options
parser = argparse.ArgumentParser(description='Train SRCDNet')

# training parameters
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--batchsize', default=1, type=int, help='batchsize')
parser.add_argument('--val_batchsize', default=1, type=int, help='batchsize for validation')
parser.add_argument('--test_batchsize', default=1, type=int, help='batchsize for test')
parser.add_argument('--num_workers', default=0, type=int, help='num_workers')
parser.add_argument('--n_class', default=2, type=int, help='number of class')
parser.add_argument('--gpu_id', default="0", type=str, help='which gpu to run.')
parser.add_argument('--suffix', default=['.tif','.png','.jpg'], type=list, help='the suffix of the image files.')
parser.add_argument('--img_size', default=256, type=int, help='batchsize for validation')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for CDNet')
parser.add_argument('--w_cd', type=float, default=0.001, help='factor to balance the weight of CD loss in Generator loss')
parser.add_argument('--scale', default=8, type=int, help='resolution difference between images. [ 2| 4| 8]')

# path for loading data from folder
parser.add_argument('--hr1_train', default='./Data/WHU_CD/train/time1', type=str, help='hr image at t1 in training set')
parser.add_argument('--lr2_train', default='./Data/WHU_CD/train/time2_lr', type=str, help='lr image at t2 in training set')
parser.add_argument('--hr2_train', default='./Data/WHU_CD/train/time2', type=str, help='hr image at t2 in training set')
parser.add_argument('--lab_train', default='./Data/WHU_CD/train/label', type=str, help='label image in training set')

parser.add_argument('--hr1_val', default='./Data/WHU_CD/val/time1', type=str, help='hr image at t1 in validation set')
parser.add_argument('--lr2_val', default='./Data/WHU_CD/val/time2_lr', type=str, help='lr image at t2 in validation set')
parser.add_argument('--hr2_val', default='./Data/WHU_CD/val/time2', type=str, help='hr image at t2 in validation set')
parser.add_argument('--lab_val', default='./Data/WHU_CD/val/label', type=str, help='label image in validation set')

parser.add_argument('--hr1_test', default='./Data/WHU_CD/test/time1', type=str, help='hr image at t1 in validation set')
parser.add_argument('--lr2_test', default='./Data/WHU_CD/test/time2_lr', type=str, help='lr image at t2 in validation set')
parser.add_argument('--hr2_test', default='./Data/WHU_CD/test/time2', type=str, help='hr image at t2 in validation set')
parser.add_argument('--lab_test', default='./Data/WHU_CD/test/label', type=str, help='label image in validation set')

# network saving and loading parameters
parser.add_argument('--model_dir', default='epochs/WHU_CD/no_spatial_attention/CD/', type=str, help='save path for CD model ')
parser.add_argument('--sr_dir', default='epochs/WHU_CD/no_spatial_attention/SR/', type=str, help='save path for Generator')
parser.add_argument('--sta_dir', default='epochs/WHU_CD/no_spatial_attention/WHU_CD.csv', type=str, help='statistics save path')

parser.add_argument('--test_model_dir', default='epochs/WHU_CD/end_to_end/CD/netCD_epoch_60.pth', type=str, help='save path for CD model ')
parser.add_argument('--test_sr_dir', default='epochs/WHU_CD/end_to_end/SR/netG_epoch_60.pth', type=str, help='save path for Generator')
