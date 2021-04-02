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


trainer = Trainer(cfg, False)

def extract(net_param, name):
	new_net_param = {}
	for param_key in net_param:
		if param_key[:len(name)] == name:
			new_net_param[param_key[len(name)+1:]] = net_param[param_key]
	return new_net_param


# This is an example code
# Please edit it if you want to save other headers
net_param = torch.load('./weights_fast_det_jig33(1.000000)_con-trg(1.000000)_tsf(0.050000)/trainer_fast_100000.pth',
						map_location=lambda storage, loc: storage)
param = extract(net_param, 'tsf_trainer')
torch.save(param, './pretrained_weights/new_tsf_head.pth')
trainer.tsf_trainer.load_state_dict(param)