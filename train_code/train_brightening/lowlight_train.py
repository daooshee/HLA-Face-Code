import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import Myloss
import numpy as np
from torchvision import transforms
import torchvision.utils as vutils

from torch.utils.tensorboard import SummaryWriter


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		try:
			m.weight.data.normal_(0.0, 0.02)
		except:
			pass
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

def update_param(optimizer):
	for param_group in optimizer.param_groups:
		param_group['lr'] = param_group['lr'] * 0.95


def train(config):
	writer = SummaryWriter(log_dir=config.log_folder)

	DCE_net = model.enhance_net_nopool().cuda()
	DCE_net.apply(weights_init)
	if config.load_pretrain == True:
		DCE_net.load_state_dict(torch.load(config.pretrain_dir))
	train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)		
	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

	L_color = Myloss.L_color()
	L_spa = Myloss.L_spa()

	L_exp = Myloss.L_exp(16,0.55)
	L_TV = Myloss.L_TV()

	optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
	DCE_net.train()

	for epoch in range(config.num_epochs):
		for iteration, img_lowlight in enumerate(train_loader):

			img_lowlight = img_lowlight.cuda()

			enhanced_image, A  = DCE_net(img_lowlight)

			Loss_TV = 200*L_TV(A)
			
			loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))

			loss_col = 5*torch.mean(L_color(enhanced_image))

			loss_exp = 10*torch.mean(L_exp(enhanced_image))
			
			
			# best_loss
			loss =  Loss_TV + loss_spa + loss_col + loss_exp
			#

			writer.add_scalars('loss', {'loss':loss}, iteration) 
			writer.add_scalars('loss', {'Loss_TV':Loss_TV}, iteration) 
			writer.add_scalars('loss', {'loss_spa':loss_spa}, iteration) 
			writer.add_scalars('loss', {'loss_col':loss_col}, iteration) 
			writer.add_scalars('loss', {'loss_exp':loss_exp}, iteration) 

			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm(DCE_net.parameters(),config.grad_clip_norm)
			optimizer.step()

			if ((iteration+1) % config.display_iter) == 0:
				print("Loss at iteration", iteration+1, ":", loss.item())

		if torch.__version__ >= '1.6.0':
			torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth', 
					   _use_new_zipfile_serialization=False)
		else:
			torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')

		vutils.save_image(img_lowlight, '{}/img_lowlight_{}.png'.format(config.log_folder, epoch), normalize=False)
		vutils.save_image(enhanced_image, '{}/enhanced_image_{}.png'.format(config.log_folder, epoch), normalize=False)

		update_param(optimizer)	

	writer.close()


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
	parser.add_argument('--lr', type=float, default=2e-4)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "snapshots/Epoch99.pth")
	parser.add_argument('--name', default='train')
	parser.add_argument('--log_folder')

	config = parser.parse_args()

	config.snapshots_folder = "snapshots_" + config.name + "/"
	config.log_folder = "logs_" + config.name + "/"


	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)

	if not os.path.exists(config.log_folder):
		os.makedirs(config.log_folder)

	train(config)