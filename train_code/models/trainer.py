#-*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torchvision.utils as vutils

from layers.modules import MultiBoxLoss
from models.factory import build_net, basenet_factory
from models.enhancement import ZeroDCE
from tools.selfsup_trainer import *


def NormBack(image):
	mean = image.new_tensor([104., 117., 123.]).view(-1, 1, 1)
	std = image.new_tensor([255, 255, 255]).view(-1, 1, 1)
	return (image+mean)/std


class Trainer(nn.Module):
	
	def __init__(self, cfg, use_cuda):
		super(Trainer, self).__init__()

		self.cfg = cfg
		self.use_cuda = use_cuda

		# Networks

		if cfg.TRAIN.MAIN:
			basenet = basenet_factory('vgg')
			self.net, feat_out_channels = build_net('train', cfg.NUM_CLASSES, 'vgg')

		if cfg.TRAIN.ROTATION:
			self.rot_trainer = BasicTrainer(feat_out_channels, 4)

		if cfg.TRAIN.CONTRASTIVE:
			self.con_trainer = ContrastiveTrainer(self.net, feat_out_channels)

		if cfg.TRAIN.JIGSAW_33:
			self.jig_33_trainer = BasicTrainer(feat_out_channels, 30)

		if cfg.TRAIN.JIGSAW_22:
			self.jig_22_trainer = BasicTrainer(feat_out_channels, 24)

		if cfg.TRAIN.JIGSAW_41:
			self.jig_41_trainer = BasicTrainer(feat_out_channels, 24)

		if cfg.TRAIN.JIGSAW_14:
			self.jig_14_trainer = BasicTrainer(feat_out_channels, 24)

		if cfg.TRAIN.CONTRASTIVE_SOURCE:
			self.consrc_trainer = ContrastiveTrainer(self.net, feat_out_channels)

		if cfg.TRAIN.CONTRASTIVE_TARGET:
			self.contrg_trainer = ContrastiveTrainer(self.net, feat_out_channels)

		if cfg.TRAIN.TRANSFER_CONTRASTIVE:
			self.tsf_trainer = ContrastiveTrainer(self.net, feat_out_channels)


	def forward(self, images, ss_data, no_det=False):

		# Detection
		
		if self.cfg.TRAIN.MAIN and not no_det:
			out = self.net(images)
		else:
			out = None

		# Self-supervised learning

		src_rot_imgs, src_rot_labels, trg_rot_imgs, trg_rot_labels, \
		src_con_im_qs, src_con_im_ks, trg_con_im_qs, trg_con_im_ks, \
		src_jig_33_imgs, src_jig_33_labels, trg_jig_33_imgs, trg_jig_33_labels, \
		src_jig_22_imgs, src_jig_22_labels, trg_jig_22_imgs, trg_jig_22_labels, \
		src_jig_41_imgs, src_jig_41_labels, trg_jig_41_imgs, trg_jig_41_labels, \
		src_jig_14_imgs, src_jig_14_labels, trg_jig_14_imgs, trg_jig_14_labels, \
		tsf_im_qs, tsf_im_ks = ss_data


		if self.use_cuda:
			loss_ss = torch.zeros([1]).cuda()
		else:
			loss_ss = torch.zeros([1])

		if self.cfg.TRAIN.ROTATION:
			src_rot_feats = self.net(src_rot_imgs, False)
			trg_rot_feats = self.net(trg_rot_imgs, False)
			
			src_rot_loss = self.rot_trainer(src_rot_feats, src_rot_labels)
			trg_rot_loss = self.rot_trainer(trg_rot_feats, trg_rot_labels)
			loss_rot = src_rot_loss + trg_rot_loss
			loss_ss = loss_ss + loss_rot * self.cfg.TRAIN_WEIGHT.ROTATION
		else:
			src_rot_loss = None
			trg_rot_loss = None
			loss_rot = None

		if self.cfg.TRAIN.JIGSAW_33:
			src_jig_33_feats = self.net(src_jig_33_imgs, False)
			trg_jig_33_feats = self.net(trg_jig_33_imgs, False)
			
			src_jig_33_loss = self.jig_33_trainer(src_jig_33_feats, src_jig_33_labels)
			trg_jig_33_loss = self.jig_33_trainer(trg_jig_33_feats, trg_jig_33_labels)
			loss_jig_33 = src_jig_33_loss + trg_jig_33_loss
			loss_ss = loss_ss + loss_jig_33 * self.cfg.TRAIN_WEIGHT.JIGSAW_33
		else:
			src_jig_33_loss = None
			trg_jig_33_loss = None
			loss_jig_33 = None

		if self.cfg.TRAIN.JIGSAW_22:
			src_jig_22_feats = self.net(src_jig_22_imgs, False)
			trg_jig_22_feats = self.net(trg_jig_22_imgs, False)
			
			src_jig_22_loss = self.jig_22_trainer(src_jig_22_feats, src_jig_22_labels)
			trg_jig_22_loss = self.jig_22_trainer(trg_jig_22_feats, trg_jig_22_labels)
			loss_jig_22 = src_jig_22_loss + trg_jig_22_loss
			loss_ss = loss_ss + loss_jig_22 * self.cfg.TRAIN_WEIGHT.JIGSAW_22
		else:
			src_jig_22_loss = None
			trg_jig_22_loss = None
			loss_jig_22 = None

		if self.cfg.TRAIN.JIGSAW_41:
			src_jig_41_feats = self.net(src_jig_41_imgs, False)
			trg_jig_41_feats = self.net(trg_jig_41_imgs, False)
			
			src_jig_41_loss = self.jig_41_trainer(src_jig_41_feats, src_jig_41_labels)
			trg_jig_41_loss = self.jig_41_trainer(trg_jig_41_feats, trg_jig_41_labels)
			loss_jig_41 = src_jig_41_loss + trg_jig_41_loss
			loss_ss = loss_ss + loss_jig_41 * self.cfg.TRAIN_WEIGHT.JIGSAW_41
		else:
			src_jig_41_loss = None
			trg_jig_41_loss = None
			loss_jig_41 = None

		if self.cfg.TRAIN.JIGSAW_14:
			src_jig_14_feats = self.net(src_jig_14_imgs, False)
			trg_jig_14_feats = self.net(trg_jig_14_imgs, False)
			
			src_jig_14_loss = self.jig_14_trainer(src_jig_14_feats, src_jig_14_labels)
			trg_jig_14_loss = self.jig_14_trainer(trg_jig_14_feats, trg_jig_14_labels)
			loss_jig_14 = src_jig_14_loss + trg_jig_14_loss
			loss_ss = loss_ss + loss_jig_14 * self.cfg.TRAIN_WEIGHT.JIGSAW_14
		else:
			src_jig_14_loss = None
			trg_jig_14_loss = None
			loss_jig_14 = None

		if self.cfg.TRAIN.CONTRASTIVE:
			src_con_feat_qs = self.net(src_con_im_qs, False)
			trg_con_feat_qs = self.net(trg_con_im_qs, False)

			self.con_trainer._momentum_update_key_encoder(self.net)

			src_con_loss = self.con_trainer(src_con_feat_qs, src_con_im_ks)
			trg_con_loss = self.con_trainer(trg_con_feat_qs, trg_con_im_ks)

			loss_con = src_con_loss + trg_con_loss
			loss_ss = loss_ss + loss_con * self.cfg.TRAIN_WEIGHT.CONTRASTIVE
		else:
			src_con_loss = None
			trg_con_loss = None
			loss_con = None

		if self.cfg.TRAIN.CONTRASTIVE_SOURCE:
			src_con_feat_qs = self.net(src_con_im_qs, False)
			self.consrc_trainer._momentum_update_key_encoder(self.net)
			loss_consrc = self.consrc_trainer(src_con_feat_qs, src_con_im_ks)

			loss_ss = loss_ss + loss_consrc * self.cfg.TRAIN_WEIGHT.CONTRASTIVE_SOURCE
		else:
			loss_consrc = None

		if self.cfg.TRAIN.CONTRASTIVE_TARGET:
			trg_con_feat_qs = self.net(trg_con_im_qs, False)
			self.contrg_trainer._momentum_update_key_encoder(self.net)
			loss_contrg = self.contrg_trainer(trg_con_feat_qs, trg_con_im_ks)

			loss_ss = loss_ss + loss_contrg * self.cfg.TRAIN_WEIGHT.CONTRASTIVE_TARGET
		else:
			loss_contrg = None

		if self.cfg.TRAIN.TRANSFER_CONTRASTIVE:
			tsf_feat_qs = self.net(tsf_im_qs, False)
			self.tsf_trainer._momentum_update_key_encoder(self.net)
			loss_tsf = self.tsf_trainer(tsf_feat_qs, tsf_im_ks)

			loss_ss = loss_ss + loss_tsf * self.cfg.TRAIN_WEIGHT.TRANSFER_CONTRASTIVE
		else:
			loss_tsf = None

		return out, loss_ss, \
				src_rot_loss, trg_rot_loss, loss_rot, \
				src_con_loss, trg_con_loss, loss_con, \
				src_jig_33_loss, trg_jig_33_loss, loss_jig_33, \
				src_jig_22_loss, trg_jig_22_loss, loss_jig_22, \
				src_jig_41_loss, trg_jig_41_loss, loss_jig_41, \
				src_jig_14_loss, trg_jig_14_loss, loss_jig_14, \
				loss_consrc, loss_contrg, loss_tsf

	def transfer_dim(self, tensor):
		tensor = tensor.unsqueeze(0).unsqueeze(0)
		return torch.cat([tensor,tensor])


def prepare_data(cfg, loader_ss_rot, loader_ss_jig_33, loader_ss_jig_22,
					loader_ss_jig_41, loader_ss_jig_14, loader_ss_consrc,
					loader_ss_contrg, loader_ss_tsf, use_cuda):

	if cfg.TRAIN.ROTATION:
		src_rot_data, trg_rot_data = loader_ss_rot()
		src_rot_imgs, src_rot_labels = src_rot_data
		trg_rot_imgs, trg_rot_labels = trg_rot_data
		
		if use_cuda: 
			src_rot_imgs = src_rot_imgs.cuda()
			src_rot_labels = src_rot_labels.cuda()
			trg_rot_imgs = trg_rot_imgs.cuda()
			trg_rot_labels = trg_rot_labels.cuda()
	else:
		src_rot_imgs = None
		src_rot_labels = None
		trg_rot_imgs = None
		trg_rot_labels = None

	if cfg.TRAIN.JIGSAW_33:
		src_jig_33_data, trg_jig_33_data = loader_ss_jig_33()
		src_jig_33_imgs, src_jig_33_labels = src_jig_33_data
		trg_jig_33_imgs, trg_jig_33_labels = trg_jig_33_data

		if use_cuda: 
			src_jig_33_imgs = src_jig_33_imgs.cuda()
			src_jig_33_labels = src_jig_33_labels.cuda()
			trg_jig_33_imgs = trg_jig_33_imgs.cuda()
			trg_jig_33_labels = trg_jig_33_labels.cuda()
	else:
		src_jig_33_imgs = None
		src_jig_33_labels = None
		trg_jig_33_imgs = None
		trg_jig_33_labels = None

	if cfg.TRAIN.JIGSAW_22:
		src_jig_22_data, trg_jig_22_data = loader_ss_jig_22()
		src_jig_22_imgs, src_jig_22_labels = src_jig_22_data
		trg_jig_22_imgs, trg_jig_22_labels = trg_jig_22_data

		if use_cuda: 
			src_jig_22_imgs = src_jig_22_imgs.cuda()
			src_jig_22_labels = src_jig_22_labels.cuda()
			trg_jig_22_imgs = trg_jig_22_imgs.cuda()
			trg_jig_22_labels = trg_jig_22_labels.cuda()
	else:
		src_jig_22_imgs = None
		src_jig_22_labels = None
		trg_jig_22_imgs = None
		trg_jig_22_labels = None

	if cfg.TRAIN.JIGSAW_41:
		src_jig_41_data, trg_jig_41_data = loader_ss_jig_41()
		src_jig_41_imgs, src_jig_41_labels = src_jig_41_data
		trg_jig_41_imgs, trg_jig_41_labels = trg_jig_41_data

		if use_cuda: 
			src_jig_41_imgs = src_jig_41_imgs.cuda()
			src_jig_41_labels = src_jig_41_labels.cuda()
			trg_jig_41_imgs = trg_jig_41_imgs.cuda()
			trg_jig_41_labels = trg_jig_41_labels.cuda()
	else:
		src_jig_41_imgs = None
		src_jig_41_labels = None
		trg_jig_41_imgs = None
		trg_jig_41_labels = None

	if cfg.TRAIN.JIGSAW_14:
		src_jig_14_data, trg_jig_14_data = loader_ss_jig_14()
		src_jig_14_imgs, src_jig_14_labels = src_jig_14_data
		trg_jig_14_imgs, trg_jig_14_labels = trg_jig_14_data

		if use_cuda: 
			src_jig_14_imgs = src_jig_14_imgs.cuda()
			src_jig_14_labels = src_jig_14_labels.cuda()
			trg_jig_14_imgs = trg_jig_14_imgs.cuda()
			trg_jig_14_labels = trg_jig_14_labels.cuda()
	else:
		src_jig_14_imgs = None
		src_jig_14_labels = None
		trg_jig_14_imgs = None
		trg_jig_14_labels = None

	if cfg.TRAIN.CONTRASTIVE or cfg.TRAIN.CONTRASTIVE_SOURCE:
		src_con_data = loader_ss_consrc()
		src_con_im_qs, src_con_im_ks = src_con_data

		if use_cuda: 
			src_con_im_qs = src_con_im_qs.cuda()
			src_con_im_ks = src_con_im_ks.cuda()
	else:
		src_con_im_qs = None
		src_con_im_ks = None

	if cfg.TRAIN.CONTRASTIVE or cfg.TRAIN.CONTRASTIVE_TARGET:
		trg_con_data = loader_ss_contrg()
		trg_con_im_qs, trg_con_im_ks = trg_con_data

		if use_cuda: 
			trg_con_im_qs = trg_con_im_qs.cuda()
			trg_con_im_ks = trg_con_im_ks.cuda()
	else:
		trg_con_im_qs = None
		trg_con_im_ks = None

	if cfg.TRAIN.TRANSFER_CONTRASTIVE:
		tsf_data = loader_ss_tsf()
		tsf_im_qs, tsf_im_ks = tsf_data

		if use_cuda: 
			tsf_im_qs = tsf_im_qs.cuda()
			tsf_im_ks = tsf_im_ks.cuda()
	else:
		tsf_im_qs = None
		tsf_im_ks = None

	return src_rot_imgs, src_rot_labels, trg_rot_imgs, trg_rot_labels, \
		src_con_im_qs, src_con_im_ks, trg_con_im_qs, trg_con_im_ks, \
		src_jig_33_imgs, src_jig_33_labels, trg_jig_33_imgs, trg_jig_33_labels, \
		src_jig_22_imgs, src_jig_22_labels, trg_jig_22_imgs, trg_jig_22_labels, \
		src_jig_41_imgs, src_jig_41_labels, trg_jig_41_imgs, trg_jig_41_labels, \
		src_jig_14_imgs, src_jig_14_labels, trg_jig_14_imgs, trg_jig_14_labels, \
		tsf_im_qs, tsf_im_ks


if __name__ == '__main__':
	from data.config import cfg
	Trainer = Trainer(cfg, use_cuda=True)