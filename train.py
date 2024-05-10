# Code for MedT

import torch
import lib
import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from albumentations.augmentations import transforms
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import torch.nn.init as init
from utils import JointTransform2D, ImageToImage2D, Image2D
from metrics import jaccard_index, f1_score, LogNLLLoss,classwise_f1
from utils import chk_mkdir, Logger, MetricList
import cv2
from functools import partial
from random import randint
import timeit
import random
from glob import glob
from sklearn.model_selection import train_test_split
from albumentations import RandomRotate90, Resize
from albumentations.core.composition import Compose, OneOf
from albumentations.augmentations import geometric


parser = argparse.ArgumentParser(description='MedT')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run(default: 400)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=8, type=int,
                    metavar='N', help='batch size (default: 1)')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--dataseed', default=2981, type=int)
parser.add_argument('--input_size', default=256, type=int)
parser.add_argument('--dataset', default='busi', type=str)
parser.add_argument('--save_freq', type=int, default = 10)

parser.add_argument('--modelname', default='MedT', type=str,
                    help='type of model')
parser.add_argument('--cuda', default="on", type=str, 
                    help='switch on/off cuda option (default: off)')
parser.add_argument('--aug', default='off', type=str,
                    help='turn on img augmentation (default: False)')
parser.add_argument('--load', default='default', type=str,
                    help='load a pretrained model')
parser.add_argument('--save', default='default', type=str,
                    help='save the model')
parser.add_argument('--direc', default='./medt', type=str,
                    help='directory to save')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--imgsize', type=int, default=None)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--gray', default='no', type=str)

args = parser.parse_args()

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

gray_ = args.gray
aug = args.aug
direc = args.direc
modelname = args.modelname
imgsize = args.input_size

if gray_ == "yes":
    from utils_gray import JointTransform2D, ImageToImage2D, Image2D
    imgchant = 1
else:
    from utils import JointTransform2D, ImageToImage2D, Image2D, UNextDataset
    imgchant = 3

if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None

data_dir = 'data'
dataset_name = args.dataset
img_ext = '.png'
if dataset_name == 'chase':
    img_ext = '.jpg'

if dataset_name == 'busi':
    mask_ext = '_mask.png'
elif dataset_name == 'glas':
    mask_ext = '.png'
elif dataset_name == 'chase':
    mask_ext = '_1stHO.png'

dataseed = args.dataseed
print('dataseed = ' + str(dataseed))
input_h = args.input_size
input_w = args.input_size
print('input_size = ' + str(args.input_size))
num_classes = 1
batch_size = 8
num_workers = 4

img_ids = sorted(
    glob(os.path.join(data_dir, dataset_name, 'images', '*' + img_ext))
)
img_ids.sort()
img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=dataseed)

train_transform = Compose([
    RandomRotate90(),
    # transforms.Flip(),
    geometric.transforms.Flip(),
    Resize(input_h, input_w),
    transforms.Normalize(),
])

val_transform = Compose([
    Resize(input_h, input_w),
    transforms.Normalize(),
])

train_dataset = UNextDataset(
    img_ids=train_img_ids,
    img_dir=os.path.join(data_dir, dataset_name, 'images'),
    mask_dir=os.path.join(data_dir, dataset_name, 'masks'),
    img_ext=img_ext,
    mask_ext=mask_ext,
    num_classes=num_classes,
    transform=train_transform)
val_dataset = UNextDataset(
    img_ids=val_img_ids,
    img_dir=os.path.join(data_dir ,dataset_name, 'images'),
    mask_dir=os.path.join(data_dir, dataset_name, 'masks'),
    img_ext=img_ext,
    mask_ext=mask_ext,
    num_classes=num_classes,
    transform=val_transform)

dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(val_dataset, 1, shuffle=True)

device = torch.device("cuda")

if modelname == "axialunet":
    model = lib.models.axialunet(img_size = imgsize, imgchan = imgchant)
elif modelname == "MedT":
    model = lib.models.axialnet.MedT(img_size = imgsize, imgchan = imgchant)
elif modelname == "gatedaxialunet":
    model = lib.models.axialnet.gated(img_size = imgsize, imgchan = imgchant)
elif modelname == "logo":
    model = lib.models.axialnet.logo(img_size = imgsize, imgchan = imgchant)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model,device_ids=[0,1]).cuda()
model.to(device)

criterion = LogNLLLoss()
optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate, weight_decay=1e-5)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))

seed = 3000
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.set_deterministic(True)
# random.seed(seed)


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)

    try:
        hd95_ = hd95(output_, target_)
    except:
        hd95_ = 0
    
    return iou, dice, hd95_


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

best_iou, best_dice, best_hd95 = 0, 0, 0
for epoch in range(args.epochs):
    epoch_running_loss = 0
    for batch_idx, (X_batch, y_batch, *rest) in enumerate(dataloader):        
    
        X_batch = Variable(X_batch.to(device ='cuda'))
        y_batch = Variable(y_batch.to(device='cuda'))
        
        # ===================forward=====================
        output = model(X_batch)
        loss = criterion(output, y_batch.squeeze().contiguous().long())
        
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_running_loss += loss.item()
        
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch, args.epochs, epoch_running_loss/(batch_idx+1)))
    
    if epoch == 10:
        for param in model.parameters():
            param.requires_grad =True
        
    if (epoch % args.save_freq) == 0:
        iou_avg_meter = AverageMeter()
        dice_avg_meter = AverageMeter()
        hd95_avg_meter = AverageMeter()

        for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
            X_batch = Variable(X_batch.to(device='cuda'))
            y_batch = Variable(y_batch.to(device='cuda'))
            # start = timeit.default_timer()
            y_out = model(X_batch)
            iou, dice, hd95_ = iou_score(y_out[:, 1, :, :], y_batch)
            iou_avg_meter.update(iou, X_batch.size(0))
            dice_avg_meter.update(dice, X_batch.size(0))
            hd95_avg_meter.update(hd95_, X_batch.size(0))

        if iou_avg_meter.avg > best_iou:
            best_iou = iou_avg_meter.avg
            best_dice = dice_avg_meter.avg
            best_hd95 = hd95_avg_meter.avg
            print('New best model')
            print('IoU: %.4f' % best_iou)
            print('Dice: %.4f' % best_dice)
            print('HD95: %.4f' % best_hd95)

            if not os.path.exists(direc):
                os.mkdir(direc)
            torch.save(model.state_dict(), os.path.join(direc, '{}.pth'.format(epoch)))

        torch.save(model.state_dict(), direc+"final_model.pth")

print('finish')
print('IoU: %.4f' % best_iou)
print('Dice: %.4f' % best_dice)
print('HD95: %.4f' % best_hd95)
