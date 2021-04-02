import torch
import torch.nn as nn
import torch.utils.data as data
from data.widerface import detection_collate
from data.widerface_rotation import WIDERDetectionRotation
from data.widerface_contrastive import WIDERDetectionContrastive
from data.widerface_jigsaw import WIDERDetectionJigsaw
from data.widerface_reconstruction import WIDERDetectionReconstruction
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.utils as vutils

from layers import *
from models.DSFD_vgg import add_extras, vgg


def NormBack(image):
	mean = image.new_tensor([104., 117., 123.]).view(-1, 1, 1)
	std = image.new_tensor([255, 255, 255]).view(-1, 1, 1)
	return (image+mean)/std


''' Data Loader '''


class SSDataloader():
	def __init__(self, batch_size, num_workers, src_file, trg_file, ss_type):
		if ss_type == 'rotation':
			src_ss_dataset = WIDERDetectionRotation(src_file)
			trg_ss_dataset = WIDERDetectionRotation(trg_file)
		elif ss_type == 'contrastive':
			src_ss_dataset = WIDERDetectionContrastive(src_file)
			trg_ss_dataset = WIDERDetectionContrastive(trg_file)
		elif ss_type == 'jigsaw_33':
			src_ss_dataset = WIDERDetectionJigsaw(src_file, '33')
			trg_ss_dataset = WIDERDetectionJigsaw(trg_file, '33')
		elif ss_type == 'jigsaw_22':
			src_ss_dataset = WIDERDetectionJigsaw(src_file, '22')
			trg_ss_dataset = WIDERDetectionJigsaw(trg_file, '22')
		elif ss_type == 'jigsaw_41':
			src_ss_dataset = WIDERDetectionJigsaw(src_file, '41')
			trg_ss_dataset = WIDERDetectionJigsaw(trg_file, '41')
		elif ss_type == 'jigsaw_14':
			src_ss_dataset = WIDERDetectionJigsaw(src_file, '14')
			trg_ss_dataset = WIDERDetectionJigsaw(trg_file, '14')
		else:
			exit('Unknown self-supervised learning type %s'%(ss_type))

		self.rot = ss_type == 'rotation'
		self.con = ss_type == 'contrastive'
		self.jig = 'jigsaw' in ss_type

		self.src_ss_loader = data.DataLoader(src_ss_dataset, batch_size,
										num_workers=num_workers,
										shuffle=True,
										pin_memory=True,
										drop_last=True)
		self.src_ss_loader_iterator = iter(self.src_ss_loader)

		self.trg_ss_loader = data.DataLoader(trg_ss_dataset, batch_size,
										num_workers=num_workers,
										shuffle=True,
										pin_memory=True,
										drop_last=True)
		self.trg_ss_loader_iterator = iter(self.trg_ss_loader)


	def __call__(self):
		try:
			src_ss_inputs = next(self.src_ss_loader_iterator)
		except:
			self.src_ss_loader_iterator = iter(self.src_ss_loader)
			src_ss_inputs = next(self.src_ss_loader_iterator)
			print('New self.src_ss_loader_iterator')

		try:
			trg_ss_inputs = next(self.trg_ss_loader_iterator)
		except:
			self.trg_ss_loader_iterator = iter(self.trg_ss_loader)
			trg_ss_inputs = next(self.trg_ss_loader_iterator)
			print('New self.trg_ss_loader_iterator')

		if self.rot or self.jig:
			src_images, src_labels = src_ss_inputs
			trg_images, trg_labels = trg_ss_inputs
			return [src_images, src_labels[:,0]], [trg_images, trg_labels[:,0]]
		
		elif self.con:
			return (src_ss_inputs, trg_ss_inputs)

		exit('Unknown self-supervised learning type')


class SSSingleDataloader():
	def __init__(self, batch_size, num_workers, file, ss_type, test=False):
		if ss_type == 'contrastive':
			ss_dataset = WIDERDetectionContrastive(file, strong=False)
		elif ss_type == 'strong_contrastive':
			ss_dataset = WIDERDetectionContrastive(file, strong=True)
		elif ss_type == 'reconstruction':
			ss_dataset = WIDERDetectionReconstruction(file)
		else:
			exit('Unknown self-supervised learning type %s'%(ss_type))

		self.ss_loader = data.DataLoader(ss_dataset, batch_size,
										num_workers=num_workers,
										shuffle=True,
										pin_memory=True,
										drop_last=True)
		self.ss_loader_iterator = iter(self.ss_loader)

	def __call__(self):
		try:
			ss_inputs = next(self.ss_loader_iterator)
		except:
			self.ss_loader_iterator = iter(self.ss_loader)
			ss_inputs = next(self.ss_loader_iterator)
			print('New self.ss_loader_iterator')

		return ss_inputs



''' Networks and Trainers '''


class UniversalMLPV1(nn.Module):
	def __init__(self, in_channel, out_channel, bottle_channel=64):
		super().__init__()
		self.conv = nn.Conv2d(in_channel, bottle_channel, 8*in_channel//1024)
		self.relu = nn.ReLU()
		self.fc = nn.Linear(bottle_channel, out_channel)
		init.kaiming_normal(self.fc.weight)
		init.kaiming_normal(self.conv.weight)

	def forward(self, x):
		x = self.relu(self.conv(x))
		x = x.view(x.size(0), x.size(1))
		return self.fc(x)


class UniversalMLPV2(nn.Module):
	def __init__(self, in_channel, out_channel, bottle_channel=64):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channel, bottle_channel, 3, padding=1)
		self.relu = nn.ReLU()
		self.conv2 = nn.Conv2d(bottle_channel, bottle_channel, 3, padding=1)
		self.fc = nn.Linear(bottle_channel, out_channel)
		init.kaiming_normal(self.fc.weight)
		init.kaiming_normal(self.conv1.weight)
		init.kaiming_normal(self.conv2.weight)

	def forward(self, x):
		x = self.relu(self.conv2(self.relu(self.conv1(x))))
		x = F.adaptive_avg_pool2d(x, (1,1))
		x = x.view(x.size(0), x.size(1))
		return self.fc(x)


class UniversalMLPV3(nn.Module):
	def __init__(self, in_channel, out_channel, bottle_channel=64):
		super().__init__()
		self.fc1 = nn.Linear((in_channel**3)//16384, bottle_channel)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(bottle_channel, out_channel)
		init.kaiming_normal(self.fc1.weight)
		init.kaiming_normal(self.fc2.weight)

	def forward(self, x):
		x = x.view(x.size(0), -1)
		return self.fc2(self.relu(self.fc1(x)))


''' Rotation '''

class BasicHead(nn.Module):
	def __init__(self, feat_out_channels, out_channel):
		super().__init__()
		self.MLP0 = UniversalMLPV2(feat_out_channels[0], out_channel)
		self.MLP1 = UniversalMLPV2(feat_out_channels[1], out_channel)
		self.MLP2 = UniversalMLPV2(feat_out_channels[2], out_channel)
		self.MLP3 = UniversalMLPV2(feat_out_channels[3], out_channel)
		self.MLP4 = UniversalMLPV2(feat_out_channels[4], out_channel)
		self.MLP5 = UniversalMLPV2(feat_out_channels[5], out_channel)

		# torch.Size([4, 256, 64, 64])
		# torch.Size([4, 512, 32, 32])
		# torch.Size([4, 512, 16, 16])
		# torch.Size([4, 1024, 8, 8])
		# torch.Size([4, 512, 4, 4])
		# torch.Size([4, 256, 2, 2])

	def forward(self, feats):
		outputs = []
		outputs.append(self.MLP0(feats[0]))
		outputs.append(self.MLP1(feats[1]))
		outputs.append(self.MLP2(feats[2]))
		outputs.append(self.MLP3(feats[3]))
		outputs.append(self.MLP4(feats[4]))
		outputs.append(self.MLP5(feats[5]))
		return outputs


class BasicTrainer(nn.Module):
	def __init__(self, feat_out_channels, class_number):
		super().__init__()
		self.Head = BasicHead(feat_out_channels, class_number)
		self.criterion = nn.CrossEntropyLoss()
		self.Weights = [1., 1., 1., 1., 1., 1.]

	def forward(self, feats, label):
		outputs = self.Head(feats)
		loss = 0.

		for output, weights in zip(outputs, self.Weights):
			loss += self.criterion(output, label) * weights
		return loss


''' Contrastive '''


class ContrastiveMLPConv(nn.Module):
	def __init__(self, in_channel, out_channel, bottle_channel=64):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channel, bottle_channel, 3, padding=1)
		self.relu = nn.ReLU()
		self.conv2 = nn.Conv2d(bottle_channel, bottle_channel, 3, padding=1)
		self.fc = nn.Linear(bottle_channel, out_channel)
		init.kaiming_normal(self.fc.weight)
		init.kaiming_normal(self.conv1.weight)
		init.kaiming_normal(self.conv2.weight)

	def forward(self, x):
		x = self.relu(self.conv2(self.relu(self.conv1(x))))
		x = F.adaptive_avg_pool2d(x, (1,1))
		x = x.view(x.size(0), x.size(1))
		return self.fc(x)


class ContrastiveMLPFC(nn.Module):
	def __init__(self, in_channel, out_channel, bottle_channel=64):
		super().__init__()
		self.fc1 = nn.Linear((in_channel**3)//16384, bottle_channel)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(bottle_channel, out_channel)
		init.kaiming_normal(self.fc1.weight)
		init.kaiming_normal(self.fc2.weight)

	def forward(self, x):
		x = x.view(x.size(0), -1)
		return self.fc2(self.relu(self.fc1(x)))


class ContrastiveHead(nn.Module):
	def __init__(self, feat_out_channels, out_channel):
		super().__init__()
		self.MLPs = []
		for in_channel in feat_out_channels:
			self.MLPs.append(ContrastiveMLPConv(in_channel, out_channel))
		self.MLPs = nn.ModuleList(self.MLPs)
	
	def forward(self, feats, bp=True):
		outputs = []
		for feat, MLP in zip(feats, self.MLPs):
			if bp:
				outputs.append(MLP(feat))
			else:
				outputs.append(MLP(feat).detach())
		return outputs


class Encoder(nn.Module):
	"""Single Shot Multibox Architecture
	The network is composed of a base VGG network followed by the
	added multibox conv layers.  Each multibox layer branches into
		1) conv2d for class conf scores
		2) conv2d for localization predictions
		3) associated priorbox layer to produce default bounding
			boxes specific to the layer's feature map size.
	See: https://arxiv.org/pdf/1512.02325.pdf for more details.

	Args:
		phase: (string) Can be "test" or "train"
		size: input image size
		base: VGG16 layers for input, size of either 300 or 500
		extras: extra layers that feed to multibox loc and conf layers
		head: "multibox head" consists of loc and conf conv layers
	"""

	def __init__(self):
		super(Encoder, self).__init__()
		vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
					512, 512, 512, 'M']
		extras_cfg = [256, 'S', 512, 128, 'S', 256]

		base = vgg(vgg_cfg, 3)
		extras = add_extras(extras_cfg, 1024)

		self.vgg = nn.ModuleList(base)
		self.L2Normof1 = L2Norm(256, 10)
		self.L2Normof2 = L2Norm(512, 8)
		self.L2Normof3 = L2Norm(512, 5)
		self.extras = nn.ModuleList(extras)

	def forward(self, x):
		pal1_sources = list()

		# apply vgg up to conv4_3 relu
		for k in range(16):
			x = self.vgg[k](x)
		of1 = x
		s = self.L2Normof1(of1)
		pal1_sources.append(s)
		
		# apply vgg up to fc7
		for k in range(16, 23):
			x = self.vgg[k](x)
		of2 = x
		s = self.L2Normof2(of2)
		pal1_sources.append(s)

		for k in range(23, 30):
			x = self.vgg[k](x)
		of3 = x
		s = self.L2Normof3(of3)
		pal1_sources.append(s)

		for k in range(30, len(self.vgg)):
			x = self.vgg[k](x)
		of4 = x
		pal1_sources.append(of4)
		
		# apply extra layers and cache source layer outputs
		for k in range(2):
			x = F.relu(self.extras[k](x), inplace=True)
		of5 = x
		pal1_sources.append(of5)
		for k in range(2, 4):
			x = F.relu(self.extras[k](x), inplace=True)
		of6 = x
		pal1_sources.append(of6)

		return pal1_sources


class ContrastiveTrainer(nn.Module):
	""" https://github.com/facebookresearch/moco/blob/master/moco/builder.py 
		关于loss越来越大: https://github.com/facebookresearch/moco/issues/9 """

	def __init__(self, encoder_q, feat_out_channels, dim=128, K=2048, m=0.99, T=0.07):
		super(ContrastiveTrainer, self).__init__()
		self.K = K
		self.m = m
		self.T = T

		self.criterion = nn.CrossEntropyLoss()
		self.encoder_k = Encoder()

		self.head_q = ContrastiveHead(feat_out_channels, dim)
		self.head_k = ContrastiveHead(feat_out_channels, dim)

		for param_k in self.encoder_k.parameters():
			param_k.requires_grad = False  # not update by gradient
		for param_k in self.head_k.parameters():
			param_k.requires_grad = False  # not update by gradient

		try:
			for param_q, param_k in zip(encoder_q.vgg.parameters(), self.encoder_k.vgg.parameters()):
				param_k.data.copy_(param_q.data)  # initialize
			for param_q, param_k in zip(encoder_q.L2Normof1.parameters(), self.encoder_k.L2Normof1.parameters()):
				param_k.data.copy_(param_q.data)  # initialize
			for param_q, param_k in zip(encoder_q.L2Normof2.parameters(), self.encoder_k.L2Normof2.parameters()):
				param_k.data.copy_(param_q.data)  # initialize
			for param_q, param_k in zip(encoder_q.L2Normof3.parameters(), self.encoder_k.L2Normof3.parameters()):
				param_k.data.copy_(param_q.data)  # initialize
			for param_q, param_k in zip(encoder_q.extras.parameters(), self.encoder_k.extras.parameters()):
				param_k.data.copy_(param_q.data)  # initialize
		except:
			for param_q, param_k in zip(encoder_q.module.vgg.parameters(), self.encoder_k.vgg.parameters()):
				param_k.data.copy_(param_q.data)  # initialize
			for param_q, param_k in zip(encoder_q.module.L2Normof1.parameters(), self.encoder_k.L2Normof1.parameters()):
				param_k.data.copy_(param_q.data)  # initialize
			for param_q, param_k in zip(encoder_q.module.L2Normof2.parameters(), self.encoder_k.L2Normof2.parameters()):
				param_k.data.copy_(param_q.data)  # initialize
			for param_q, param_k in zip(encoder_q.module.L2Normof3.parameters(), self.encoder_k.L2Normof3.parameters()):
				param_k.data.copy_(param_q.data)  # initialize
			for param_q, param_k in zip(encoder_q.module.extras.parameters(), self.encoder_k.extras.parameters()):
				param_k.data.copy_(param_q.data)  # initialize
		for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
			param_k.data.copy_(param_q.data)  # initialize

		self.register_buffer("queue", torch.randn(len(feat_out_channels), dim, K))
		self.queue = nn.functional.normalize(self.queue, dim=1)
		self.register_buffer("queue_ptr", torch.zeros(len(feat_out_channels), dtype=torch.long))

	@torch.no_grad()
	def _momentum_update_key_encoder(self, encoder_q):
		"""
		Momentum update of the key encoder
		"""
		for param_q, param_k in zip(encoder_q.vgg.parameters(), self.encoder_k.vgg.parameters()):
			param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
		for param_q, param_k in zip(encoder_q.L2Normof1.parameters(), self.encoder_k.L2Normof1.parameters()):
			param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
		for param_q, param_k in zip(encoder_q.L2Normof2.parameters(), self.encoder_k.L2Normof2.parameters()):
			param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
		for param_q, param_k in zip(encoder_q.L2Normof3.parameters(), self.encoder_k.L2Normof3.parameters()):
			param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
		for param_q, param_k in zip(encoder_q.extras.parameters(), self.encoder_k.extras.parameters()):
			param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

		# for param_q, param_k in zip(encoder_q.module.vgg.parameters(), self.encoder_k.vgg.parameters()):
		# 	param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
		# for param_q, param_k in zip(encoder_q.module.L2Normof1.parameters(), self.encoder_k.L2Normof1.parameters()):
		# 	param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
		# for param_q, param_k in zip(encoder_q.module.L2Normof2.parameters(), self.encoder_k.L2Normof2.parameters()):
		# 	param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
		# for param_q, param_k in zip(encoder_q.module.L2Normof3.parameters(), self.encoder_k.L2Normof3.parameters()):
		# 	param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
		# for param_q, param_k in zip(encoder_q.module.extras.parameters(), self.encoder_k.extras.parameters()):
		# 	param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

		for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
			param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


	@torch.no_grad()
	def _dequeue_and_enqueue(self, keys, ind):
		# gather keys before updating queue
		keys = concat_all_gather(keys)

		batch_size = keys.shape[0]

		ptr = int(self.queue_ptr[ind])
		assert self.K % batch_size == 0  # for simplicity

		# replace the keys at ptr (dequeue and enqueue)
		self.queue[ind, :, ptr:ptr + batch_size] = keys.T
		ptr = (ptr + batch_size) % self.K  # move pointer

		self.queue_ptr[ind] = ptr

	@torch.no_grad()
	def _batch_shuffle_ddp(self, x):
		"""
		Batch shuffle, for making use of BatchNorm.
		*** Only support DistributedDataParallel (DDP) model. ***
		"""
		# gather from all gpus
		batch_size_this = x.shape[0]
		x_gather = concat_all_gather(x)
		batch_size_all = x_gather.shape[0]

		num_gpus = batch_size_all // batch_size_this

		# random shuffle index
		idx_shuffle = torch.randperm(batch_size_all).cuda()

		# broadcast to all gpus
		torch.distributed.broadcast(idx_shuffle, src=0)

		# index for restoring
		idx_unshuffle = torch.argsort(idx_shuffle)

		# shuffled index for this gpu
		gpu_idx = torch.distributed.get_rank()
		idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

		return x_gather[idx_this], idx_unshuffle

	@torch.no_grad()
	def _batch_unshuffle_ddp(self, x, idx_unshuffle):
		"""
		Undo batch shuffle.
		*** Only support DistributedDataParallel (DDP) model. ***
		"""
		# gather from all gpus
		batch_size_this = x.shape[0]
		x_gather = concat_all_gather(x)
		batch_size_all = x_gather.shape[0]

		num_gpus = batch_size_all // batch_size_this

		# restored index for this gpu
		gpu_idx = torch.distributed.get_rank()
		idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

		return x_gather[idx_this]

	def NCELoss(self, q, k, ind):
		q = nn.functional.normalize(q, dim=1)
		with torch.no_grad():
			k = nn.functional.normalize(k, dim=1)

		# compute logits
		# Einstein sum is more intuitive
		# positive logits: Nx1
		l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
		# negative logits: NxK
		l_neg = torch.einsum('nc,ck->nk', [q, self.queue[ind].clone().detach()])

		# logits: Nx(1+K)
		logits = torch.cat([l_pos, l_neg], dim=1)

		# apply temperature
		logits /= self.T

		# labels: positive key indicators
		labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

		# dequeue and enqueue
		self._dequeue_and_enqueue(k, ind)

		return self.criterion(logits, labels)

	def forward(self, feats_q, im_k):
		# print('encoder_q',net.vgg[0].weight[0,0,0,:])
		# print('encoder_k',self.encoder_k.vgg[0].weight[0,0,0,:])
		# print('head_q',self.head_q.MLPs[0].conv1.weight[0,0,0,:]) 
		# print('head_k',self.head_k.MLPs[0].conv1.weight[0,0,0,:])

		# compute query features
		q = self.head_q(feats_q)  # queries: NxC

		# compute key features
		with torch.no_grad():  # no gradient to keys
			
			# shuffle for making use of BN
			im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

			k = self.head_k(self.encoder_k(im_k))  # keys: NxC

			# undo shuffle
			for i in range(len(q)):
				k[i] = self._batch_unshuffle_ddp(k[i], idx_unshuffle)

		loss = 0.
		for i in range(len(q)):
			loss += self.NCELoss(q[i],k[i],i)
			
		return loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
	"""
	Performs all_gather operation on the provided tensors.
	*** Warning ***: torch.distributed.all_gather has no gradient.
	"""
	tensors_gather = [torch.ones_like(tensor)
		for _ in range(torch.distributed.get_world_size())]
	torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

	output = torch.cat(tensors_gather, dim=0)
	return output





class UniversalMLPV2(nn.Module):
	def __init__(self, in_channel, out_channel, bottle_channel=64):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channel, bottle_channel, 3, padding=1)
		self.relu = nn.ReLU()
		self.conv2 = nn.Conv2d(bottle_channel, bottle_channel, 3, padding=1)
		self.fc = nn.Linear(bottle_channel, out_channel)
		init.kaiming_normal(self.fc.weight)
		init.kaiming_normal(self.conv1.weight)
		init.kaiming_normal(self.conv2.weight)

	def forward(self, x):
		x = self.relu(self.conv2(self.relu(self.conv1(x))))
		x = F.adaptive_avg_pool2d(x, (1,1))
		x = x.view(x.size(0), x.size(1))
		return self.fc(x)


''' Reconstruct '''

class FeatToImg(nn.Module):
	def __init__(self, in_channel, out_channel, bottle_channel=32):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channel, bottle_channel, 3, padding=1)
		self.conv2 = nn.Conv2d(bottle_channel, bottle_channel, 3, padding=1)
		self.conv3 = nn.Conv2d(bottle_channel, out_channel, 1)
		self.relu = nn.ReLU()
		
		init.kaiming_normal(self.conv1.weight)
		init.kaiming_normal(self.conv2.weight)
		init.kaiming_normal(self.conv3.weight)

	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		return self.conv3(x)

class RecHead(nn.Module):
	def __init__(self, feat_out_channels, out_channel):
		super().__init__()
		self.MLP0 = FeatToImg(feat_out_channels[0], out_channel)
		self.MLP1 = FeatToImg(feat_out_channels[1], out_channel)
		self.MLP2 = FeatToImg(feat_out_channels[2], out_channel)
		self.MLP3 = FeatToImg(feat_out_channels[3], out_channel)
		self.MLP4 = FeatToImg(feat_out_channels[4], out_channel)
		self.MLP5 = FeatToImg(feat_out_channels[5], out_channel)

		# torch.Size([4, 256, 64, 64])
		# torch.Size([4, 512, 32, 32])
		# torch.Size([4, 512, 16, 16])
		# torch.Size([4, 1024, 8, 8])
		# torch.Size([4, 512, 4, 4])
		# torch.Size([4, 256, 2, 2])

	def forward(self, feats):
		outputs = []
		outputs.append(self.MLP0(feats[0]))
		outputs.append(self.MLP1(feats[1]))
		outputs.append(self.MLP2(feats[2]))
		outputs.append(self.MLP3(feats[3]))
		outputs.append(self.MLP4(feats[4]))
		outputs.append(self.MLP5(feats[5]))
		return outputs


class RecTrainer(nn.Module):
	def __init__(self, feat_out_channels):
		super().__init__()
		self.Head = RecHead(feat_out_channels, 3)
		self.Weights = [1., 1., 1., 1., 1., 1.]

	def forward(self, feats, label):
		outputs = self.Head(feats)
		loss = 0.

		for output, weights in zip(outputs, self.Weights):
			label = F.interpolate(label, size=(output.shape[2], output.shape[3]))
			loss += torch.mean(torch.abs(output - label)) * weights
		return loss