#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import time
import glob
import torch
from tqdm import tqdm
import argparse
import pandas as pd
import random

import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from data.config import cfg
from layers.modules import MultiBoxLoss
from data.widerface import WIDERDetection, detection_collate
from models.factory import build_net, basenet_factory
from models.enhancement import *
from trainer import Trainer, prepare_data
from tools.selfsup_trainer import *

parser = argparse.ArgumentParser(
    description='DSFD face Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--batch_size',
                    default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--model',
                    default='vgg', type=str,
                    choices=['vgg', 'resnet50', 'resnet101', 'resnet152'],
                    help='model for training')
parser.add_argument('--resume',
                    default='weights/wider_face_pretrained.pth', type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers',
                    default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda',
                    default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',
                    default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--multigpu',
                    default=True, type=bool,
                    help='Use mutil Gpu training')
parser.add_argument('--save_folder',
                    default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        cudnn.benckmark = True
        if not args.multigpu:
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        else:
            import torch.distributed as dist
            dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)

    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")


val_dataset = WIDERDetection(cfg.FACE.SRC_VAL_FILE, mode='val')
val_loader = data.DataLoader(val_dataset, args.batch_size,
                             num_workers=args.num_workers,
                             shuffle=False,
                             collate_fn=detection_collate,
                             pin_memory=True)

if cfg.TRAIN.ROTATION:
    val_loader_ss_rot = SSDataloader(args.batch_size, args.num_workers, 
              cfg.FACE.SRC_VAL_FILE, cfg.FACE.TAR_VAL_FILE, 'rotation')
else:
    val_loader_ss_rot = None

if cfg.TRAIN.JIGSAW_33:
    val_loader_ss_jig_33 = SSDataloader(args.batch_size, args.num_workers, 
              cfg.FACE.SRC_VAL_FILE, cfg.FACE.TAR_VAL_FILE, 'jigsaw_33')
else:
    val_loader_ss_jig_33 = None

if cfg.TRAIN.JIGSAW_22:
    val_loader_ss_jig_22 = SSDataloader(args.batch_size, args.num_workers, 
              cfg.FACE.SRC_VAL_FILE, cfg.FACE.TAR_VAL_FILE, 'jigsaw_22')
else:
    val_loader_ss_jig_22 = None

if cfg.TRAIN.JIGSAW_41:
    val_loader_ss_jig_41 = SSDataloader(args.batch_size, args.num_workers, 
              cfg.FACE.SRC_VAL_FILE, cfg.FACE.TAR_VAL_FILE, 'jigsaw_41')
else:
    val_loader_ss_jig_41 = None

if cfg.TRAIN.JIGSAW_14:
    val_loader_ss_jig_14 = SSDataloader(args.batch_size, args.num_workers, 
              cfg.FACE.SRC_VAL_FILE, cfg.FACE.TAR_VAL_FILE, 'jigsaw_14')
else:
    val_loader_ss_jig_14 = None

if cfg.TRAIN.CONTRASTIVE_SOURCE or cfg.TRAIN.CONTRASTIVE:
    val_loader_ss_consrc = SSSingleDataloader(args.batch_size, args.num_workers, 
              cfg.FACE.SRC_VAL_FILE, 'contrastive')
else:
    val_loader_ss_consrc = None

if cfg.TRAIN.CONTRASTIVE_TARGET or cfg.TRAIN.CONTRASTIVE:
    val_loader_ss_contrg = SSSingleDataloader(args.batch_size, args.num_workers, 
              cfg.FACE.TAR_VAL_FILE, 'contrastive')
else:
    val_loader_ss_contrg = None

if cfg.TRAIN.RECONSTRUCTION:
    val_loader_ss_rec = SSSingleDataloader(args.batch_size, args.num_workers, 
              cfg.FACE.TAR_VAL_FILE, 'reconstruction')
else:
    val_loader_ss_rec = None

if cfg.TRAIN.TRANSFER_CONTRASTIVE:
    val_loader_ss_tsf = SSSingleDataloader(args.batch_size, args.num_workers, 
              cfg.FACE.SRC_VAL_FILE, 'transfer', test=True)
else:
    val_loader_ss_tsf = None








DataList = {'Checkpoint':[], 'l_pal2':[], 'c_pal2':[], 'total_2':[]}




trainer = Trainer(cfg, args.cuda)
trainer.eval()

if args.cuda:
    trainer = trainer.cuda()

criterion = MultiBoxLoss(cfg)


def val(checkpoint):
    trainer.load_state_dict(torch.load(checkpoint))

    if args.multigpu:
        trainer_final = torch.nn.DataParallel(trainer)
    else:
        trainer_final = trainer


    step = 0
    total_loss_l_pal2 = 0
    total_loss_c_pal2 = 0
    total_loss_ss = 0

    for (images, targets) in tqdm(val_loader):
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]

        ss_data = prepare_data(cfg, val_loader_ss_rot, 
                                    val_loader_ss_jig_33, \
                                    val_loader_ss_jig_22, \
                                    val_loader_ss_jig_41, \
                                    val_loader_ss_jig_14, \
                                    val_loader_ss_consrc, \
                                    val_loader_ss_contrg, \
                                    val_loader_ss_rec, \
                                    val_loader_ss_tsf, args.cuda)

        out, loss_ss = trainer_final(images, ss_data)[:2]
        loss_l_pal2, loss_c_pal2 = criterion(out[3:], targets)
        
        total_loss_l_pal2 += loss_l_pal2.mean().item()
        total_loss_c_pal2 += loss_c_pal2.mean().item()
        total_loss_ss += loss_ss.mean().item()

        step += 1

    # total_loss_l_pal1 = total_loss_l_pal1 / step
    # total_loss_c_pal1 = total_loss_c_pal1 / step
    total_loss_l_pal2 = total_loss_l_pal2 / step
    total_loss_c_pal2 = total_loss_c_pal2 / step
    total_loss_ss = total_loss_ss / step

    print('%s, %.6f, %.6f, %.6f, %.6f, %.6f' % (checkpoint, total_loss_l_pal2, total_loss_c_pal2, total_loss_l_pal2+total_loss_c_pal2, total_loss_ss, total_loss_l_pal2+total_loss_c_pal2 + total_loss_ss))

    # DataList['Checkpoint'].append(checkpoint)
    # DataList['l_pal2'].append(total_loss_l_pal2)
    # DataList['c_pal2'].append(total_loss_c_pal2)
    # DataList['total_2'].append(total_loss_l_pal2+total_loss_c_pal2)


with torch.no_grad():
    List = glob.glob('./weights/vgg_det_con-src(0.050000)/trainer_*.pth')[10:]
    print(List)
    for checkpoint in List:
        val(checkpoint)

# df = pd.DataFrame(DataList)
# df = df.sort_values(by=['c_pal2'])
# print(df)
