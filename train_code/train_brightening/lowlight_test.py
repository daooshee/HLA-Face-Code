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
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import cv2

DCE_net = model.enhance_net_nopool().cuda()
DCE_net.load_state_dict(torch.load('./Illumination-Enhancer.pth'))

def lowlight(image_path):
	result_path = image_path.replace('images','images_enhanced')
	print(image_path, '-->', result_path)

	data_lowlight = Image.open(image_path)
	data_lowlight = data_lowlight.convert('RGB')
	data_lowlight = (np.asarray(data_lowlight)/255.0)
	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)

	data_lowlight = data_lowlight.unsqueeze(0).cuda()

	enhanced_image, r = DCE_net(data_lowlight)

	torchvision.utils.save_image(enhanced_image, result_path)

with torch.no_grad():
	file_list = glob.glob('../dataset/DarkFace/images/train/*.*')
	for image in file_list:
		lowlight(image)