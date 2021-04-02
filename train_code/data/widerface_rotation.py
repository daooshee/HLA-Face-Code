#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from PIL import Image, ImageDraw
import torch.utils.data as data
import numpy as np
import random
from utils.augmentations import process_rotation_multiscale

def tensor_rot_90(x):
    return x.flip(2).transpose(1,2)

def tensor_rot_180(x):
    return x.flip(2).flip(1)

def tensor_rot_270(x):
    return x.transpose(1,2).flip(2)


class WIDERDetectionRotation(data.Dataset):
    def __init__(self, list_file):
        super(WIDERDetectionRotation, self).__init__()
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
        self.size = 256

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

                label = np.random.randint(4)
                if label == 1:
                    img = img.rotate(90)
                elif label == 2:
                    img = img.rotate(180)
                elif label == 3:
                    img = img.rotate(270)

                img = process_rotation_multiscale(img)
                break
                
            except Exception as e:
                print('Error in WIDERDetectionRotation:', image_path, e)
                index = random.randrange(0, self.num_samples)

        return torch.from_numpy(img), torch.from_numpy(np.ones([1])*label).long()

    def annotransform(self, boxes, im_width, im_height):
        boxes[:, 0] /= im_width
        boxes[:, 1] /= im_height
        boxes[:, 2] /= im_width
        boxes[:, 3] /= im_height
        return boxes