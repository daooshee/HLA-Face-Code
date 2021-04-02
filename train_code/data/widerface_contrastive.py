#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from PIL import Image, ImageDraw, ImageFilter
import torch.utils.data as data
import numpy as np
import random
from utils.augmentations import transform_to_dsfd
import torchvision.transforms as transforms

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class WIDERDetectionContrastive(data.Dataset):
    def __init__(self, list_file):
        super(WIDERDetectionContrastive, self).__init__()
        self.fnames = []
        self.boxes = []
        self.labels = []

        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split()
            num_faces = int(line[1])
            box = []
            label = []
            for i in range(num_faces):
                x = float(line[2 + 5 * i])
                y = float(line[3 + 5 * i])
                w = float(line[4 + 5 * i])
                h = float(line[5 + 5 * i])
                c = int(line[6 + 5 * i])
                if w <= 0 or h <= 0:
                    continue
                box.append([x, y, x + w, y + h])
                label.append(c)
            if len(box) > 0:
                self.fnames.append(line[0])
                self.boxes.append(box)
                self.labels.append(label)

        self.num_samples = len(self.boxes)
        self.size = 288

        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        while True:
            try:
                image_path = self.fnames[index]
                img = Image.open(image_path)
                if img.mode == 'L':
                    img = img.convert('RGB')

                im_width, im_height = img.size

                # select a face
                target_face = random.choice(self.boxes[index])
                left_x = (target_face[0]+target_face[2]) // 2 - self.size // 2 + random.randint(-16,16)
                left_y = (target_face[1]+target_face[3]) // 2 - self.size // 2 + random.randint(-16,16)

                left_x = np.clip(left_x, 0, im_width-self.size)
                left_y = np.clip(left_y, 0, im_height-self.size)

                img = img.crop([left_x, left_y, left_x+self.size, left_y+self.size])

                x1 = transform_to_dsfd(self.augmentation(img))
                x2 = transform_to_dsfd(self.augmentation(img))

                return torch.from_numpy(x1), torch.from_numpy(x2)
    
            except Exception as e:
                print('Error in WIDERDetectionContrastive:', image_path, e)
                index = random.randrange(0, self.num_samples)
