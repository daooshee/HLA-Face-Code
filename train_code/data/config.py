#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
from easydict import EasyDict
import numpy as np


_C = EasyDict()
cfg = _C

# data augument config
_C.expand_prob = 0.5
_C.expand_max_ratio = 4
_C.hue_prob = 0.5
_C.hue_delta = 18
_C.contrast_prob = 0.5
_C.contrast_delta = 0.5
_C.saturation_prob = 0.5
_C.saturation_delta = 0.5
_C.brightness_prob = 0.5
_C.brightness_delta = 0.125
_C.data_anchor_sampling_prob = 0.5
_C.min_face_size = 6.0
_C.apply_distort = True
_C.apply_expand = False
_C.img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype(
    'float32')
_C.resize_width = 640
_C.resize_height = 640
_C.scale = 1 / 127.0
_C.anchor_sampling = True
_C.filter_min_face = True

# train config
_C.LR_STEPS = (20000, 100000)
_C.MAX_STEPS = 70000
_C.EPOCHES = 100000

# anchor config
_C.FEATURE_MAPS = [160, 80, 40, 20, 10, 5]
_C.INPUT_SIZE = 640
_C.STEPS = [4, 8, 16, 32, 64, 128]
_C.ANCHOR_SIZES1 = [8, 16, 32, 64, 128, 256]
_C.ANCHOR_SIZES2 = [16, 32, 64, 128, 256, 512]
_C.ASPECT_RATIO = [1.0]
_C.CLIP = False
_C.VARIANCE = [0.1, 0.2]

# detection config
_C.NMS_THRESH = 0.3
_C.NMS_TOP_K = 5000
_C.TOP_K = 750
_C.CONF_THRESH = 0.05

# loss config
_C.NEG_POS_RATIOS = 3
_C.NUM_CLASSES = 2

# face config
_C.FACE = EasyDict()
_C.FACE.SRC_TRAIN_FILE = './dataset/wider_face_train.txt'
_C.FACE.SRC_VAL_FILE = './dataset/wider_face_val.txt'
_C.FACE.TAR_TRAIN_FILE = './dataset/mf_dsfd_dark_face_train.txt'
_C.FACE.OVERLAP_THRESH = 0.35

# training setting
_C.TRAIN = EasyDict()
_C.TRAIN.PRE_PROCESS = False
_C.TRAIN.MAIN = True
_C.TRAIN.ROTATION = False
_C.TRAIN.JIGSAW_33 = True
_C.TRAIN.JIGSAW_22 = False
_C.TRAIN.JIGSAW_41 = False
_C.TRAIN.JIGSAW_14 = False
_C.TRAIN.CONTRASTIVE = False
_C.TRAIN.CONTRASTIVE_SOURCE = False
_C.TRAIN.CONTRASTIVE_TARGET = True
_C.TRAIN.TRANSFER_CONTRASTIVE = True

_C.TRAIN_WEIGHT = EasyDict()
_C.TRAIN_WEIGHT.ROTATION = 0.5
_C.TRAIN_WEIGHT.JIGSAW_33 = 0.075
_C.TRAIN_WEIGHT.JIGSAW_22 = 0.05
_C.TRAIN_WEIGHT.JIGSAW_41 = 0.05
_C.TRAIN_WEIGHT.JIGSAW_14 = 0.05
_C.TRAIN_WEIGHT.CONTRASTIVE = 0.01
_C.TRAIN_WEIGHT.CONTRASTIVE_SOURCE = 0.05
_C.TRAIN_WEIGHT.CONTRASTIVE_TARGET = 0.05
_C.TRAIN_WEIGHT.TRANSFER_CONTRASTIVE = 0.05