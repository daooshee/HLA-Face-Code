#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import time
import numpy as np
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from data.config import cfg
from data.widerface import WIDERDetection, detection_collate
from models.trainer import Trainer, prepare_data
from tools.selfsup_trainer import *


parser = argparse.ArgumentParser(
	description='DSFD face Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
# ------------------------------------------------------------------------------------------------
parser.add_argument('--batch_size',
					default=8, type=int,
					help='Batch size for training')
parser.add_argument('--model',
					default='vgg', type=str,
					choices=['vgg', 'resnet50', 'resnet101', 'resnet152'],
					help='model for training')
parser.add_argument('--num_workers',
					default=4, type=int,
					help='Number of workers used in dataloading')
parser.add_argument('--save_folder',
					default='weights/',
					help='Directory for saving checkpoint models')
# ------------------------------------------------------------------------------------------------
parser.add_argument('--cuda',
					default=True, type=bool,
					help='Use CUDA to train model')
parser.add_argument('--multigpu',
					action='store_true', help='Use mutil Gpu training')
# ------------------------------------------------------------------------------------------------
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
					help='Weight decay for SGD')
parser.add_argument('--gamma',
					default=0.1, type=float,
					help='Gamma update for SGD')
# ------------------------------------------------------------------------------------------------
parser.add_argument('--continue_training', 
					action='store_true', help='continue to train from the last best checkpoint')
parser.add_argument('--continue_training_iteration',
					type=int, help='Checkpoint to load')
parser.add_argument('--debug', action='store_true', help='debug mode')
# ------------------------------------------------------------------------------------------------
args = parser.parse_args()




''' Cuda Settings '''

if torch.cuda.is_available():
	if args.cuda:
		torch.backends.cudnn.benckmark = True

		if not args.multigpu:
			os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
		else:
			import torch.distributed as dist
			dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)

	if not args.cuda:
		print("WARNING: It looks like you have a CUDA device, but aren't " +
			  "using CUDA.\nRun with --cuda for optimal training speed.")
		torch.set_default_tensor_type('torch.FloatTensor')




''' Pathes and Names '''

save_folder = os.path.join(args.save_folder, args.model)
debug_folder = './debug'
debug_folder += args.save_folder[7:].replace('/','')

if cfg.TRAIN.MAIN:
	save_folder += '_det'
	debug_folder += '_det'
if cfg.TRAIN.PRE_PROCESS:
	save_folder += '_dceFixed'
	debug_folder += '_dceFixed'
if cfg.TRAIN.ROTATION:
	save_folder += '_rot(%f)'%(cfg.TRAIN_WEIGHT.ROTATION)
	debug_folder += '_rot(%f)'%(cfg.TRAIN_WEIGHT.ROTATION)
if cfg.TRAIN.JIGSAW_33:
	save_folder += '_jig33(%f)'%(cfg.TRAIN_WEIGHT.JIGSAW_33)
	debug_folder += '_jig33(%f)'%(cfg.TRAIN_WEIGHT.JIGSAW_33)
if cfg.TRAIN.JIGSAW_22:
	save_folder += '_jig22(%f)'%(cfg.TRAIN_WEIGHT.JIGSAW_22)
	debug_folder += '_jig22(%f)'%(cfg.TRAIN_WEIGHT.JIGSAW_22)
if cfg.TRAIN.JIGSAW_41:
	save_folder += '_jig41(%f)'%(cfg.TRAIN_WEIGHT.JIGSAW_41)
	debug_folder += '_jig41(%f)'%(cfg.TRAIN_WEIGHT.JIGSAW_41)
if cfg.TRAIN.JIGSAW_14:
	save_folder += '_jig14(%f)'%(cfg.TRAIN_WEIGHT.JIGSAW_14)
	debug_folder += '_jig14(%f)'%(cfg.TRAIN_WEIGHT.JIGSAW_14)
if cfg.TRAIN.CONTRASTIVE:
	save_folder += '_con(%f)'%(cfg.TRAIN_WEIGHT.CONTRASTIVE)
	debug_folder += '_con(%f)'%(cfg.TRAIN_WEIGHT.CONTRASTIVE)
	assert args.multigpu
if cfg.TRAIN.CONTRASTIVE_SOURCE:
	save_folder += '_con-src(%f)'%(cfg.TRAIN_WEIGHT.CONTRASTIVE_SOURCE)
	debug_folder += '_con-src(%f)'%(cfg.TRAIN_WEIGHT.CONTRASTIVE_SOURCE)
	assert args.multigpu
if cfg.TRAIN.CONTRASTIVE_TARGET:
	save_folder += '_con-trg(%f)'%(cfg.TRAIN_WEIGHT.CONTRASTIVE_TARGET)
	debug_folder += '_con-trg(%f)'%(cfg.TRAIN_WEIGHT.CONTRASTIVE_TARGET)
	assert args.multigpu
if cfg.TRAIN.TRANSFER_CONTRASTIVE:
	save_folder += '_tsf(%f)'%(cfg.TRAIN_WEIGHT.TRANSFER_CONTRASTIVE)
	debug_folder += '_tsf(%f)'%(cfg.TRAIN_WEIGHT.TRANSFER_CONTRASTIVE)
	assert args.multigpu

if not os.path.exists(save_folder):
	os.makedirs(save_folder)

if not os.path.exists(debug_folder):
	os.makedirs(debug_folder)

writer = SummaryWriter(log_dir=debug_folder)




''' Datasets '''

train_dataset = WIDERDetection(cfg.FACE.SRC_TRAIN_FILE, mode='train')

train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size,
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
trainer.train()

net_param = torch.load('./pretrained_weights/dsfd_vgg_0.880.pth', map_location=lambda storage, loc: storage)
trainer.net.load_state_dict(net_param)




''' Load fast pre-trained headers '''

if cfg.TRAIN.ROTATION:
	net_param = torch.load('./pretrained_weights/rot_head.pth', map_location=lambda storage, loc: storage)
	trainer.rot_trainer.load_state_dict(net_param)

if cfg.TRAIN.JIGSAW_33:
	net_param = torch.load('./pretrained_weights/jig_33_head.pth', map_location=lambda storage, loc: storage)
	trainer.jig_33_trainer.load_state_dict(net_param)

if cfg.TRAIN.JIGSAW_22:
	net_param = torch.load('./pretrained_weights/jig_22_head.pth', map_location=lambda storage, loc: storage)
	trainer.jig_22_trainer.load_state_dict(net_param)

if cfg.TRAIN.JIGSAW_41:
	net_param = torch.load('./pretrained_weights/jig_41_head.pth', map_location=lambda storage, loc: storage)
	trainer.jig_41_trainer.load_state_dict(net_param)

if cfg.TRAIN.JIGSAW_14:
	net_param = torch.load('./pretrained_weights/jig_14_head.pth', map_location=lambda storage, loc: storage)
	trainer.jig_14_trainer.load_state_dict(net_param)

if cfg.TRAIN.CONTRASTIVE:
	net_param = torch.load('./pretrained_weights/con_head.pth', map_location=lambda storage, loc: storage)
	trainer.con_trainer.load_state_dict(net_param)

if cfg.TRAIN.CONTRASTIVE_SOURCE:
	net_param = torch.load('./pretrained_weights/consrc_head.pth', map_location=lambda storage, loc: storage)
	trainer.consrc_trainer.load_state_dict(net_param)

if cfg.TRAIN.CONTRASTIVE_TARGET:
	net_param = torch.load('./pretrained_weights/contrg_head.pth', map_location=lambda storage, loc: storage)
	trainer.contrg_trainer.load_state_dict(net_param)

if cfg.TRAIN.TRANSFER_CONTRASTIVE:
	net_param = torch.load('./pretrained_weights/tsf_head.pth', map_location=lambda storage, loc: storage)
	trainer.tsf_trainer.load_state_dict(net_param)

criterion = MultiBoxLoss(cfg)

if args.continue_training:
	file = 'trainer_{}.pth'.format(args.continue_training_iteration)
	trainer.load_state_dict(torch.load(os.path.join(save_folder, file)))

if args.cuda:
	trainer = trainer.cuda()
	if args.multigpu:
		trainer = torch.nn.DataParallel(trainer)




''' Optimizer '''


class Schedular():
	def __init__(self, optimizer, gamma, start_iteration=0, warmup_iter=1000):
		self.iteration = start_iteration
		self.warmup_iter = warmup_iter
		self.lr = args.warmuplr
		self.step_index = 0
		self.gamma = gamma

		if start_iteration > warmup_iter:
			for step in cfg.LR_STEPS:
				if start_iteration > step:
					self.step_index += 1
			self.lr = args.lr * (self.gamma ** (self.step_index))
		else:
			self.lr = args.warmuplr + (args.lr-args.warmuplr) / warmup_iter * start_iteration
		
		self.update_param(optimizer)

	def update(self, optimizer):
		self.iteration += 1
		if self.iteration < self.warmup_iter:
			self.lr = args.warmuplr + (args.lr-args.warmuplr) / self.warmup_iter * self.iteration
			self.update_param(optimizer)
		elif self.iteration in cfg.LR_STEPS:
			self.step_index += 1
			self.lr = args.lr * (self.gamma ** (self.step_index))
			self.update_param(optimizer)
		return self.lr

	def update_param(self, optimizer):
		for param_group in optimizer.param_groups:
			param_group['lr'] = self.lr

optimizer = torch.optim.SGD(trainer.parameters(), lr=args.warmuplr, momentum=args.momentum, weight_decay=args.weight_decay)

start_epoch = 0
iteration = 0

if args.continue_training:
	file = 'optimizer_{}.pth'.format(args.continue_training_iteration)
	optimizer.load_state_dict(torch.load(os.path.join(save_folder, file)))
	iteration = args.continue_training_iteration

schedular = Schedular(optimizer, args.gamma, iteration)




''' Main Training '''


def merge_multigpu(tensor):
	if tensor is not None:
		if len(tensor.shape) == 0:
			return tensor
		return tensor.sum() / len(tensor)
	else:
		return None


for epoch in range(start_epoch, cfg.EPOCHES):
	losses = 0

	for batch_idx, (images, targets) in enumerate(train_loader):
		optimizer.zero_grad()
		t0 = time.time()

		''' Load Data '''

		if args.cuda:
			images = images.cuda()
			targets = [ann.cuda() for ann in targets]
		else:
			images = images
			targets = [ann for ann in targets]

		ss_data = prepare_data(cfg, train_loader_ss_rot, 
									train_loader_ss_jig_33, \
									train_loader_ss_jig_22, \
									train_loader_ss_jig_41, \
									train_loader_ss_jig_14, \
									train_loader_ss_consrc, \
									train_loader_ss_contrg, \
									train_loader_ss_tsf, args.cuda)

		''' Forward '''

		if (iteration % 100 == 0 and iteration > 0) or args.debug:
			out, loss_ss, \
				src_rot_loss, trg_rot_loss, loss_rot, \
				src_con_loss, trg_con_loss, loss_con, \
				src_jig_33_loss, trg_jig_33_loss, loss_jig_33, \
				src_jig_22_loss, trg_jig_22_loss, loss_jig_22, \
				src_jig_41_loss, trg_jig_41_loss, loss_jig_41, \
				src_jig_14_loss, trg_jig_14_loss, loss_jig_14, \
				loss_consrc, loss_contrg, loss_tsf = trainer(images, ss_data)
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
			out, loss_ss = trainer(images, ss_data)[:2]
			loss_ss = merge_multigpu(loss_ss)

		loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targets)
		loss_l_pa12, loss_c_pal2 = criterion(out[3:], targets)
		loss_det = loss_l_pa1l + loss_c_pal1 + loss_l_pa12 + loss_c_pal2

		loss = loss_det + loss_ss
		loss = loss

		''' Backward '''

		loss.backward()
		optimizer.step()

		lr = schedular.update(optimizer)
		t1 = time.time()
		losses += loss.item()

		''' Display '''

		if (iteration % 10 == 0 and iteration > 0) or args.debug:
			print('Timer: %.4f' % (t1 - t0))
			print('epoch:{} (best:{}) || iter:{} || Lr:{}'.format(epoch, best_epoch, iteration, lr))
			print('Loss:{:.4f} || det:{:.4f} || ss:{:.4f}'.format(loss, loss_det, loss_ss))
			print('->>lr:{}'.format(optimizer.param_groups[0]['lr']))
			writer.add_scalars('scalar/total_loss', {'total':loss}, iteration) 
			writer.add_scalars('scalar/lr', {'lr':lr}, iteration)

			if (iteration % 100 == 0 and iteration > 0) or args.debug:

				if cfg.TRAIN.MAIN:
					print('->> loss detection:{:.4f}'.format(loss_det))
					print('    pal1 conf loss:{:.4f} || pal1 loc loss:{:.4f}'.format(
						loss_c_pal1, loss_l_pa1l))
					print('    pal2 conf loss:{:.4f} || pal2 loc loss:{:.4f}'.format(
						loss_c_pal2, loss_l_pa12))

				if cfg.TRAIN.ROTATION:
					print('->> loss rotation (not used in Paper):{:.4f}, src:{:.4f} || trg:{:.4f}'.format(
						loss_rot, src_rot_loss, trg_rot_loss))

				if cfg.TRAIN.CONTRASTIVE:
					print('->> loss contrastive (not used in Paper):{:.4f}, src:{:.4f} || trg:{:.4f}'.format(
						loss_con, src_con_loss, trg_con_loss))

				if cfg.TRAIN.JIGSAW_33:
					print('->> loss jigsaw 33 (E(L) <=> H):{:.4f}, src:{:.4f} || trg:{:.4f}'.format(
						loss_jig_33, src_jig_33_loss, trg_jig_33_loss))

				if cfg.TRAIN.JIGSAW_22:
					print('->> loss jigsaw 22 (not used in Paper):{:.4f}, src:{:.4f} || trg:{:.4f}'.format(
						loss_jig_22, src_jig_22_loss, trg_jig_22_loss))

				if cfg.TRAIN.JIGSAW_41:
					print('->> loss jigsaw 41 (not used in Paper):{:.4f}, src:{:.4f} || trg:{:.4f}'.format(
						loss_jig_41, src_jig_41_loss, trg_jig_41_loss))

				if cfg.TRAIN.JIGSAW_14:
					print('->> loss jigsaw 14 (not used in Paper):{:.4f}, src:{:.4f} || trg:{:.4f}'.format(
						loss_jig_14, src_jig_14_loss, trg_jig_14_loss))

				if cfg.TRAIN.CONTRASTIVE_SOURCE:
					print('->> loss contrastive source (not used in Paper):{:.4f}'.format(loss_consrc))

				if cfg.TRAIN.CONTRASTIVE_TARGET:
					print('->> loss contrastive target (E(L) up):{:.4f}'.format(loss_contrg))

				if cfg.TRAIN.TRANSFER_CONTRASTIVE:
					print('->> loss joint-D contrastive source (H <=> D(H)):{:.4f}'.format(loss_tsf))

				loss_dicts = {}

				if cfg.TRAIN.MAIN:
					loss_dicts['c_pal1'] = loss_c_pal1
					loss_dicts['l_pa1l'] = loss_l_pa1l
					loss_dicts['c_pal2'] = loss_c_pal2
					loss_dicts['l_pa12'] = loss_l_pa12
					loss_dicts['det'] = loss_det

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

				writer.add_scalars('scalar/loss', loss_dicts, iteration)

		if (iteration % 2500 == 0 and iteration > 0) or args.debug:
			print('Saving state, iter:', iteration)
			file = 'trainer_' + repr(iteration) + '.pth'

			if torch.__version__ >= '1.6.0':
				if args.multigpu:
					torch.save(trainer.module.state_dict(),
							os.path.join(save_folder, file), _use_new_zipfile_serialization=False)
				else:
					torch.save(trainer.state_dict(),
							os.path.join(save_folder, file), _use_new_zipfile_serialization=False)
				opt_file = 'optimizer_' + repr(iteration) + '.pth'
				torch.save(optimizer.state_dict(),
						os.path.join(save_folder, opt_file), _use_new_zipfile_serialization=False)
			else:
				if args.multigpu:
					torch.save(trainer.module.state_dict(),
							os.path.join(save_folder, file))
				else:
					torch.save(trainer.state_dict(),
							os.path.join(save_folder, file))
				opt_file = 'optimizer_' + repr(iteration) + '.pth'
				torch.save(optimizer.state_dict(),
						os.path.join(save_folder, opt_file))
		iteration += 1
		
		if args.debug:
			break

	# val(epoch)

	if args.debug:
		exit('\nSuccess!')

	if iteration >= cfg.MAX_STEPS:
		break

writer.close()
