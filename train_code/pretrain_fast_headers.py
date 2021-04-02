#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from data.config import cfg
from data.widerface import WIDERDetection, detection_collate
from models.trainer import Trainer, prepare_data
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
parser.add_argument('--num_workers',
					default=4, type=int,
					help='Number of workers used in dataloading')
parser.add_argument('--cuda',
					default=True, type=bool,
					help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate',
					default=1e-4, type=float,
					help='initial learning rate')
parser.add_argument('--warmuplr', 
					default=1e-6, type=float,
					help='warm up initial learning rate')
parser.add_argument('--momentum',
					default=0.9, type=float,
					help='Momentum value for optim')
parser.add_argument('--weight_decay',
					default=5e-4, type=float,
					help='Weight decay for Adam')
parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--multigpu',
					action='store_true', help='Use mutil Gpu training')
parser.add_argument('--save_folder',
					default='weights_fast',
					help='Directory for saving checkpoint models')
parser.add_argument('--load_fast_train', 
					action='store_true', help='skip fast train and load checkpoints')
parser.add_argument('--continue_training', 
					action='store_true', help='continue to train from the last best checkpoint')
parser.add_argument('--fast_pretrain_iter',
					default=100000, type=int,
					help='Iterations of pretraining self-supervised learning heads')
parser.add_argument('--debug', action='store_true', help='debug mode')

args = parser.parse_args()


''' Cuda Settings '''

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
		torch.set_default_tensor_type('torch.FloatTensor')


cfg.TRAIN_WEIGHT.ROTATION = 1
cfg.TRAIN_WEIGHT.JIGSAW_33 = 1
cfg.TRAIN_WEIGHT.JIGSAW_22 = 1
cfg.TRAIN_WEIGHT.JIGSAW_41 = 1
cfg.TRAIN_WEIGHT.JIGSAW_14 = 1
cfg.TRAIN_WEIGHT.CONTRASTIVE = 1
cfg.TRAIN_WEIGHT.CONTRASTIVE_SOURCE= 1
cfg.TRAIN_WEIGHT.CONTRASTIVE_TARGET= 1

''' Pathes and Names '''

debug_folder = './debug'
debug_folder += args.save_folder[7:].replace('/','')

if cfg.TRAIN.MAIN:
	args.save_folder += '_det'
	debug_folder += '_det'
if cfg.TRAIN.PRE_PROCESS:
	args.save_folder += '_dceFixed'
	debug_folder += '_dceFixed'
if cfg.TRAIN.ROTATION:
	args.save_folder += '_rot(%f)'%(cfg.TRAIN_WEIGHT.ROTATION)
	debug_folder += '_rot(%f)'%(cfg.TRAIN_WEIGHT.ROTATION)
if cfg.TRAIN.JIGSAW_33:
	args.save_folder += '_jig33(%f)'%(cfg.TRAIN_WEIGHT.JIGSAW_33)
	debug_folder += '_jig33(%f)'%(cfg.TRAIN_WEIGHT.JIGSAW_33)
if cfg.TRAIN.JIGSAW_22:
	args.save_folder += '_jig22(%f)'%(cfg.TRAIN_WEIGHT.JIGSAW_22)
	debug_folder += '_jig22(%f)'%(cfg.TRAIN_WEIGHT.JIGSAW_22)
if cfg.TRAIN.JIGSAW_41:
	args.save_folder += '_jig41(%f)'%(cfg.TRAIN_WEIGHT.JIGSAW_41)
	debug_folder += '_jig41(%f)'%(cfg.TRAIN_WEIGHT.JIGSAW_41)
if cfg.TRAIN.JIGSAW_14:
	args.save_folder += '_jig14(%f)'%(cfg.TRAIN_WEIGHT.JIGSAW_14)
	debug_folder += '_jig14(%f)'%(cfg.TRAIN_WEIGHT.JIGSAW_14)
if cfg.TRAIN.CONTRASTIVE:
	args.save_folder += '_con(%f)'%(cfg.TRAIN_WEIGHT.CONTRASTIVE)
	debug_folder += '_con(%f)'%(cfg.TRAIN_WEIGHT.CONTRASTIVE)
	assert args.multigpu
if cfg.TRAIN.CONTRASTIVE_SOURCE:
	sargs.ave_folder += '_con-src(%f)'%(cfg.TRAIN_WEIGHT.CONTRASTIVE_SOURCE)
	debug_folder += '_con-src(%f)'%(cfg.TRAIN_WEIGHT.CONTRASTIVE_SOURCE)
	assert args.multigpu
if cfg.TRAIN.CONTRASTIVE_TARGET:
	args.save_folder += '_con-trg(%f)'%(cfg.TRAIN_WEIGHT.CONTRASTIVE_TARGET)
	debug_folder += '_con-trg(%f)'%(cfg.TRAIN_WEIGHT.CONTRASTIVE_TARGET)
	assert args.multigpu
if cfg.TRAIN.TRANSFER_CONTRASTIVE:
	args.save_folder += '_tsf(%f)'%(cfg.TRAIN_WEIGHT.TRANSFER_CONTRASTIVE)
	debug_folder += '_tsf(%f)'%(cfg.TRAIN_WEIGHT.TRANSFER_CONTRASTIVE)
	assert args.multigpu

if not os.path.exists(args.save_folder):
	os.makedirs(args.save_folder)

if not os.path.exists(debug_folder):
	os.makedirs(debug_folder)

writer = SummaryWriter(log_dir=debug_folder)


''' Datasets '''

# val_batchsize = args.batch_size // 2

train_dataset = WIDERDetection(cfg.FACE.SRC_TRAIN_FILE, mode='train')
# val_dataset = WIDERDetection(cfg.FACE.SRC_VAL_FILE, mode='val')

train_loader = data.DataLoader(train_dataset, args.batch_size,
								num_workers=args.num_workers,
								shuffle=True,
								collate_fn=detection_collate,
								pin_memory=True)


if cfg.TRAIN.ROTATION:
	train_loader_ss_rot = SSDataloader(args.batch_size, args.num_workers, 
				cfg.FACE.SRC_TRAIN_FILE, cfg.FACE.TAR_TRAIN_FILE, 'rotation')
else:
	train_loader_ss_rot = None


if cfg.TRAIN.JIGSAW_33:
	train_loader_ss_jig_33 = SSDataloader(args.batch_size, args.num_workers, 
				cfg.FACE.SRC_TRAIN_FILE, cfg.FACE.TAR_TRAIN_FILE, 'jigsaw_33')
else:
	train_loader_ss_jig_33 = None

if cfg.TRAIN.JIGSAW_22:
	train_loader_ss_jig_22 = SSDataloader(args.batch_size, args.num_workers, 
				cfg.FACE.SRC_TRAIN_FILE, cfg.FACE.TAR_TRAIN_FILE, 'jigsaw_22')
else:
	train_loader_ss_jig_22 = None

if cfg.TRAIN.JIGSAW_41:
	train_loader_ss_jig_41 = SSDataloader(args.batch_size, args.num_workers, 
				cfg.FACE.SRC_TRAIN_FILE, cfg.FACE.TAR_TRAIN_FILE, 'jigsaw_41')
else:
	train_loader_ss_jig_41 = None

if cfg.TRAIN.JIGSAW_14:
	train_loader_ss_jig_14 = SSDataloader(args.batch_size, args.num_workers, 
				cfg.FACE.SRC_TRAIN_FILE, cfg.FACE.TAR_TRAIN_FILE, 'jigsaw_14')
else:
	train_loader_ss_jig_14 = None

if cfg.TRAIN.CONTRASTIVE_SOURCE or cfg.TRAIN.CONTRASTIVE:
	train_loader_ss_consrc = SSSingleDataloader(args.batch_size, args.num_workers, 
				cfg.FACE.SRC_TRAIN_FILE, 'contrastive')
else:
	train_loader_ss_consrc = None

if cfg.TRAIN.CONTRASTIVE_TARGET or cfg.TRAIN.CONTRASTIVE:
	train_loader_ss_contrg = SSSingleDataloader(args.batch_size, args.num_workers, 
				cfg.FACE.TAR_TRAIN_FILE, 'contrastive')
else:
	train_loader_ss_contrg = None

if cfg.TRAIN.TRANSFER_CONTRASTIVE:
	train_loader_ss_tsf = SSSingleDataloader(args.batch_size, args.num_workers, 
				cfg.FACE.SRC_TRAIN_FILE, 'transfer')
else:
	train_loader_ss_tsf = None


''' Tools '''


min_loss = np.inf
best_epoch = -1

def NormBack(image):
	mean = image.new_tensor([104., 117., 123.]).view(-1, 1, 1)
	std = image.new_tensor([255, 255, 255]).view(-1, 1, 1)
	return (image+mean)/std


''' Models '''

trainer = Trainer(cfg, args.cuda)
if args.cuda:
	trainer = trainer.cuda()
trainer.train()

''' Fast train for self-supervised learning headers, without multi-gpu '''

def merge_multigpu(tensor):
	if tensor is not None:
		if len(tensor.shape) == 0:
			return tensor
		return tensor.sum() / len(tensor)
	else:
		return None

def update_param(optimizer):
	for param_group in optimizer.param_groups:
		param_group['lr'] = param_group['lr'] * 0.5

parameters = []
if cfg.TRAIN.ROTATION:
	parameters += [{'params': trainer.rot_trainer.parameters()}]
if cfg.TRAIN.CONTRASTIVE:
	parameters += [{'params': trainer.con_trainer.parameters()}]
if cfg.TRAIN.JIGSAW_33:
	parameters += [{'params': trainer.jig_33_trainer.parameters()}]
if cfg.TRAIN.JIGSAW_22:
	parameters += [{'params': trainer.jig_22_trainer.parameters()}]
if cfg.TRAIN.JIGSAW_41:
	parameters += [{'params': trainer.jig_41_trainer.parameters()}]
if cfg.TRAIN.JIGSAW_14:
	parameters += [{'params': trainer.jig_14_trainer.parameters()}]
if cfg.TRAIN.CONTRASTIVE_SOURCE:
	parameters += [{'params': trainer.consrc_trainer.parameters()}]
if cfg.TRAIN.CONTRASTIVE_TARGET:
	parameters += [{'params': trainer.contrg_trainer.parameters()}]
if cfg.TRAIN.TRANSFER_CONTRASTIVE:
	parameters += [{'params': trainer.tsf_trainer.parameters()}]
fast_optimizer = optim.Adam(parameters, lr=1e-4)

if args.multigpu:
	trainer = torch.nn.DataParallel(trainer)

for i in range(args.fast_pretrain_iter):
	fast_optimizer.zero_grad()

	# Load Data
	ss_data = prepare_data(cfg, train_loader_ss_rot, 
								train_loader_ss_jig_33, \
								train_loader_ss_jig_22, \
								train_loader_ss_jig_41, \
								train_loader_ss_jig_14, \
								train_loader_ss_consrc, \
								train_loader_ss_contrg, \
								train_loader_ss_tsf, args.cuda)


	# Forward
	if (i % 100 == 0 and i > 0) or args.debug:
		_, loss_ss, \
			src_rot_loss, trg_rot_loss, loss_rot, \
			src_con_loss, trg_con_loss, loss_con, \
			src_jig_33_loss, trg_jig_33_loss, loss_jig_33, \
			src_jig_22_loss, trg_jig_22_loss, loss_jig_22, \
			src_jig_41_loss, trg_jig_41_loss, loss_jig_41, \
			src_jig_14_loss, trg_jig_14_loss, loss_jig_14, \
			loss_consrc, loss_contrg, loss_tsf = trainer(None, ss_data, no_det=True)
		loss_ss = merge_multigpu(loss_ss)
		loss_rot = merge_multigpu(loss_rot)
		src_rot_loss = merge_multigpu(src_rot_loss)
		trg_rot_loss = merge_multigpu(trg_rot_loss)
		loss_con = merge_multigpu(loss_con)
		src_con_loss = merge_multigpu(src_con_loss)
		trg_con_loss = merge_multigpu(trg_con_loss)
		loss_jig_33 = merge_multigpu(loss_jig_33)
		loss_jig_22 = merge_multigpu(loss_jig_22)
		loss_jig_41 = merge_multigpu(loss_jig_41)
		loss_jig_14 = merge_multigpu(loss_jig_14)
		src_jig_33_loss = merge_multigpu(src_jig_33_loss)
		trg_jig_33_loss = merge_multigpu(trg_jig_33_loss)
		src_jig_22_loss = merge_multigpu(src_jig_22_loss)
		trg_jig_22_loss = merge_multigpu(trg_jig_22_loss)
		src_jig_41_loss = merge_multigpu(src_jig_41_loss)
		trg_jig_41_loss = merge_multigpu(trg_jig_41_loss)
		src_jig_14_loss = merge_multigpu(src_jig_14_loss)
		trg_jig_14_loss = merge_multigpu(trg_jig_14_loss)
		loss_consrc = merge_multigpu(loss_consrc)
		loss_contrg = merge_multigpu(loss_contrg)
		loss_tsf = merge_multigpu(loss_tsf)
	else:
		loss_ss = trainer(None, ss_data, no_det=True)[1]
		loss_ss = merge_multigpu(loss_ss)

	# Backward
	loss_ss.backward()
	fast_optimizer.step()

	if (i % 10 == 0 and i > 0) or args.debug:

		print('Iteration:{} || loss_ss:{:.4f}'.format(i,loss_ss))
		print('->> lr:{}'.format(fast_optimizer.param_groups[0]['lr']))

		if i % 100 == 0 or args.debug:

			if cfg.TRAIN.ROTATION:
				print('->> loss rotation:{:.4f}, src:{:.4f} || trg:{:.4f}'.format(
					loss_rot, src_rot_loss, trg_rot_loss))

			if cfg.TRAIN.CONTRASTIVE:
				print('->> loss contrastive:{:.4f}, src:{:.4f} || trg:{:.4f}'.format(
					loss_con, src_con_loss, trg_con_loss))

			if cfg.TRAIN.JIGSAW_33:
				print('->> loss jigsaw 33:{:.4f}, src:{:.4f} || trg:{:.4f}'.format(
					loss_jig_33, src_jig_33_loss, trg_jig_33_loss))

			if cfg.TRAIN.JIGSAW_22:
				print('->> loss jigsaw 22:{:.4f}, src:{:.4f} || trg:{:.4f}'.format(
					loss_jig_22, src_jig_22_loss, trg_jig_22_loss))

			if cfg.TRAIN.JIGSAW_41:
				print('->> loss jigsaw 41:{:.4f}, src:{:.4f} || trg:{:.4f}'.format(
					loss_jig_41, src_jig_41_loss, trg_jig_41_loss))

			if cfg.TRAIN.JIGSAW_14:
				print('->> loss jigsaw 14:{:.4f}, src:{:.4f} || trg:{:.4f}'.format(
					loss_jig_14, src_jig_14_loss, trg_jig_14_loss))

			if cfg.TRAIN.CONTRASTIVE_SOURCE:
				print('->> loss contrastive source:{:.4f}'.format(loss_consrc))

			if cfg.TRAIN.CONTRASTIVE_TARGET:
				print('->> loss contrastive target:{:.4f}'.format(loss_contrg))

			if cfg.TRAIN.TRANSFER_CONTRASTIVE:
				print('->> loss transfer contrastive target:{:.4f}'.format(loss_tsf))

			loss_dicts = {}

			if cfg.TRAIN.ROTATION:
				loss_dicts['src_rot'] = src_rot_loss
				loss_dicts['trg_rot'] = trg_rot_loss
				loss_dicts['rot'] = loss_rot

			if cfg.TRAIN.CONTRASTIVE:
				loss_dicts['src_con'] = src_con_loss
				loss_dicts['trg_con'] = trg_con_loss
				loss_dicts['con'] = loss_con

			if cfg.TRAIN.JIGSAW_33:
				loss_dicts['src_jig_33'] = src_jig_33_loss
				loss_dicts['trg_jig_33'] = trg_jig_33_loss
				loss_dicts['jig_33'] = loss_jig_33

			if cfg.TRAIN.JIGSAW_22:
				loss_dicts['src_jig_22'] = src_jig_22_loss
				loss_dicts['trg_jig_22'] = trg_jig_22_loss
				loss_dicts['jig_22'] = loss_jig_22

			if cfg.TRAIN.JIGSAW_41:
				loss_dicts['src_jig_41'] = src_jig_41_loss
				loss_dicts['trg_jig_41'] = trg_jig_41_loss
				loss_dicts['jig_41'] = loss_jig_41

			if cfg.TRAIN.JIGSAW_14:
				loss_dicts['src_jig_14'] = src_jig_14_loss
				loss_dicts['trg_jig_14'] = trg_jig_14_loss
				loss_dicts['jig_14'] = loss_jig_14

			if cfg.TRAIN.CONTRASTIVE_SOURCE:
				loss_dicts['consrc'] = loss_consrc

			if cfg.TRAIN.CONTRASTIVE_TARGET:
				loss_dicts['contrg'] = loss_contrg

			if cfg.TRAIN.TRANSFER_CONTRASTIVE:
					loss_dicts['tsf'] = loss_tsf

			writer.add_scalars('scalar/loss', loss_dicts, i)

			if i % 10000 == 0 or args.debug:
				update_param(fast_optimizer)

				file = 'trainer_fast_' + repr(i) + '.pth'
				if args.multigpu:
					torch.save(trainer.module.state_dict(), 
							   os.path.join(args.save_folder, file),
							   _use_new_zipfile_serialization=False)
				else:
					torch.save(trainer.state_dict(),
							   os.path.join(args.save_folder, file),
							   _use_new_zipfile_serialization=False)

	if args.debug:
		break

writer.close()