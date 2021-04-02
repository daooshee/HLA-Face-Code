#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from PIL import Image, ImageDraw
import torch.utils.data as data
import numpy as np
import random
from utils.augmentations import transform_to_dsfd
import glob


class ImageLoader(data.Dataset):
    def __init__(self, path="/mnt/hdd/wangwenjing/Dataset/DarkFace/"):
        super(ImageLoader, self).__init__()
        self.ori_fnames = glob.glob(path+"images/train/*.png")
        self.num_samples = len(self.ori_fnames)

    def __len__(self):
        return self.num_samples

    def load_image(self, img_path):

        return 

    def __getitem__(self, index):
        ori_image_path = self.ori_fnames[index]
        ori_img = Image.open(ori_image_path)
        if ori_img.mode == 'L':
            ori_img = ori_img.convert('RGB')

        rand_x = random.randint(0, ori_img.size[0]-256)
        rand_y = random.randint(0, ori_img.size[1]-256)
        ori_img = ori_img.crop([rand_x,rand_y,rand_x+256,rand_y+256])

        ori_img = transform_to_dsfd(ori_img)
        return torch.from_numpy(ori_img)



