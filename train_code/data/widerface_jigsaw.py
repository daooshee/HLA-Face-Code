#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from PIL import Image, ImageDraw
import torch.utils.data as data
import numpy as np
import random
from utils.augmentations import process_jigsaw
import cv2


permutations_4 = [[0, 1, 2, 3],
                [0, 1, 3, 2],
                [0, 2, 1, 3],
                [0, 2, 3, 1],
                [0, 3, 1, 2],
                [0, 3, 2, 1],
                [1, 0, 2, 3],
                [1, 0, 3, 2],
                [1, 2, 0, 3],
                [1, 2, 3, 0],
                [1, 3, 0, 2],
                [1, 3, 2, 0],
                [2, 0, 1, 3],
                [2, 0, 3, 1],
                [2, 1, 0, 3],
                [2, 1, 3, 0],
                [2, 3, 0, 1],
                [2, 3, 1, 0],
                [3, 0, 1, 2],
                [3, 0, 2, 1],
                [3, 1, 0, 2],
                [3, 1, 2, 0],
                [3, 2, 0, 1],
                [3, 2, 1, 0]]


class WIDERDetectionJigsaw(data.Dataset):
    def __init__(self, list_file, jigsaw_type):
        super(WIDERDetectionJigsaw, self).__init__()
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.jigsaw_type = jigsaw_type

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

        if jigsaw_type == '33':
            self.permutations = self.__retrive_permutations()
            self.size = 255
            self.crop_positions = []
            for i in range(3):
                for j in range(3):
                    self.crop_positions.append([i*85,j*85,i*85+85,j*85+85])
        else:
            self.permutations = permutations_4
            self.size = 256
            self.crop_positions = []
            if jigsaw_type == '22':
                for i in range(2):
                    for j in range(2):
                        self.crop_positions.append([i*128,j*128,i*128+128,j*128+128])
            elif jigsaw_type == '41':
                for i in range(4):
                    self.crop_positions.append([i*64,0,i*64+64,256])
            elif jigsaw_type == '14':
                for i in range(4):
                    self.crop_positions.append([0,i*64,256,i*64+64,])

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
                left_x = (target_face[0]+target_face[2]) // 2 - 256 // 2 + random.randint(-16,16)
                left_y = (target_face[1]+target_face[3]) // 2 - 256 // 2 + random.randint(-16,16)

                left_x = np.clip(left_x, 0, im_width-256)
                left_y = np.clip(left_y, 0, im_height-256)

                img = img.crop([left_x, left_y, left_x+256, left_y+256])

                new_size = random.randint(256,640)
                img = img.resize((new_size, new_size))
                edge = new_size - self.size
                img = img.crop([edge,edge,edge+self.size,edge+self.size])

                if self.jigsaw_type == '33':
                    data, order = self.jigsaw_33(img)
                if self.jigsaw_type == '22':
                    data, order = self.jigsaw_22(img)
                if self.jigsaw_type == '41':
                    data, order = self.jigsaw_41(img)
                if self.jigsaw_type == '14':
                    data, order = self.jigsaw_14(img)

                return data, torch.from_numpy(np.ones([1])*order).long()
                
            except Exception as e:
                print('Error in WIDERDetectionJigsaw:', image_path, e)
                index = random.randrange(0, self.num_samples)

    def __retrive_permutations(self):
        all_perm = np.load('data/permutations_hamming_max_30.npy')
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm

    def jigsaw_33(self, img):
        tiles = [None] * 9
        # jigsaw
        for n in range(9):
            tile = img.crop(self.crop_positions[n])
            tile = process_jigsaw(tile)
            tiles[n] = torch.from_numpy(tile)

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(9)]

        row1 = torch.cat(data[0:3], 1)
        row2 = torch.cat(data[3:6], 1)
        row3 = torch.cat(data[6:9], 1)
        data = torch.cat([row1, row2, row3], 2)
        return data, order

    def jigsaw_22(self, img):
        tiles = [None] * 4
        # jigsaw
        for n in range(4):
            tile = img.crop(self.crop_positions[n])
            tile = process_jigsaw(tile, edge=32)
            tiles[n] = torch.from_numpy(tile)

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(4)]

        row1 = torch.cat(data[0:2], 1)
        row2 = torch.cat(data[2:4], 1)
        data = torch.cat([row1, row2], 2)
        return data, order

    def jigsaw_41(self, img):
        tiles = [None] * 4
        # jigsaw
        for n in range(4):
            tile = img.crop(self.crop_positions[n])
            random_edge = random.randint(0,48)
            tile = tile.crop([0,random_edge,64,random_edge+208])
            tile = process_jigsaw(tile, edge=16)
            tiles[n] = torch.from_numpy(tile)

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(4)]

        data = torch.cat(data, 2)
        return data, order

    def jigsaw_14(self, img):
        tiles = [None] * 4
        # jigsaw
        for n in range(4):
            tile = img.crop(self.crop_positions[n])
            random_edge = random.randint(0,48)
            tile = tile.crop([random_edge,0,random_edge+208,64])
            tile = process_jigsaw(tile, edge=16)
            tiles[n] = torch.from_numpy(tile)

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(4)]

        data = torch.cat(data, 1)
        return data, order
