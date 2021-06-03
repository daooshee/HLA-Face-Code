# Training Code for HLA-Face: Joint High-Low Adaptation for Low Light Face Detection

The official PyTorch implementation **(training code)** for HLA-Face: Joint High-Low Adaptation for Low Light Face Detection (CVPR21).

You can find more information on our [project website](https://daooshee.github.io/HLA-Face-Website/).

------



## 1. Requirements

- Python 3
- PyTorch (I use version 1.60. I think other versions would also be OK)
- opencv
- numpy
- easydict



## 2. Data preparation

### **2.1 DARK FACE**

Download the DARK FACE training and validation images (DarkFace_Train.zip) from https://flyywh.github.io/CVPRW2019LowLight/.

Only the ./images folder is needed, the labels are not used. Finally, organize images as

```
./dataset/DarkFace/images/train/xxx.png
./dataset/DarkFace/images/train/xxx.png
./dataset/DarkFace/images/train/xxx.png
```

### **2.2 WIDER FACE**

Download the WIDER FACE dataset from http://shuoyang1213.me/WIDERFACE/, and organize images as 

```
./dataset/WiderFace/WIDER_train/images/x-xxx/xxx.jpg
./dataset/WiderFace/WIDER_train/images/x-xxx/xxx.jpg
./dataset/WiderFace/WIDER_train/images/x-xxx/xxx.jpg

./dataset/WiderFace/WIDER_val/images/x-xxx/xxx.jpg
./dataset/WiderFace/WIDER_val/images/x-xxx/xxx.jpg
./dataset/WiderFace/WIDER_val/images/x-xxx/xxx.jpg
```

Different from the original face annotations, we reorganize them into

```
./dataset/wider_face_train.txt
./dataset/wider_face_val.txt
```



## 3. Training brightening  E(L)

Please refer to [./train_brightening](./train_brightening/README.md).

If you want to skip the training process, the trained checkpoint can be found in `./train_brightening/Illumination-Enhancer.pth`.

If you want to skip the whole brightening process, the enhanced DARK FACE training set can be downloaded from [[Google]](https://drive.google.com/drive/folders/1m82GForByEYnRiFt5GyLq-EHn2PHoolr?usp=sharing) [[Baidu (vua4)]](https://pan.baidu.com/s/175YEtaXSmAEHazkXmsr_Xg).



## 4. Training noise synthesis H_noise

Please refer to [./train_noise_synthesis](./train_noise_synthesis/README.md).

If you want to skip this step, the distorted WIDER FACE training set can be downloaded from [[Google]](https://drive.google.com/drive/folders/1LHlPPG1MkSY9QE6lVudDpdc3eTebwg6w?usp=sharing) [[Baidu (j7b7)]](https://pan.baidu.com/s/1b2Ybgc5h2rKoc_e2l_WQMA).



## 5. Joint High-Low Adaptation

### **5.1 Dataset preparation**

In summary, three datasets are needed for this process.

- Original WIDER FACE training set (from step 2.2)

```
./dataset/WiderFace/WIDER_train/images/x-xxx/xxx.jpg
./dataset/WiderFace/WIDER_train/images/x-xxx/xxx.jpg
./dataset/WiderFace/WIDER_train/images/x-xxx/xxx.jpg
```

- Enhanced DARK FACE training set (from step 3)

```
./dataset/DarkFace/images_enhanced/train/xxx.png
./dataset/DarkFace/images_enhanced/train/xxx.png
./dataset/DarkFace/images_enhanced/train/xxx.png
```

- Distorted WIDER FACE training set (from step 4)

```
./dataset/WiderFace/WIDER_images_add_noise/train/xxx.jpg
./dataset/WiderFace/WIDER_images_add_noise/train/xxx.jpg
./dataset/WiderFace/WIDER_images_add_noise/train/xxx.jpg
```



To avoid randomly cropping too many meaningless patches, we first use [MF](https://github.com/baidut/BIMEF/blob/master/lowlight/mf.m) + [DSFD](https://github.com/yxlijun/DSFD.pytorch) to detect the faces in DARK FACE. If you want to skip this step, the detected results have been provided in `./dataset/mf_dsfd_dark_face_train.txt`.



### **5.2 Pre-training headers**

Before training the whole framework, we need to pre-train the self-supervised learning headers

```
python pretrain_fast_headers.py
```

Then transfer the checkpoints

```
python transfer_headers.py
```

Note that `transfer_headers.py` is only an example script. Please edit it to save specific headers.



If you want to skip this step, pre-trained headers can be downloaded from [[Google]](https://drive.google.com/drive/folders/1INhVq2XNcfyz8w2D71KP5Ucmhtmc9m01?usp=sharing) [[Baidu (jhgg)]](https://pan.baidu.com/s/1lNsxf9YY-k0CmNH4JN6hdg).

Finally, organize the pre-trained headers as

```
./pretrained_weights/jig_33_head.pth
./pretrained_weights/tsf_head.pth
./pretrained_weights/contrg_head.pth
```



### **5.3 Pre-training on WIDER FACE**

The face detector DSFD is first pre-trained on WIDER FACE. We use the checkpoint provided by https://github.com/yxlijun/DSFD.pytorch. Please download and save it to `./pretrained_weights/dsfd_vgg_0.880.pth`.

For non Baidu users, the checkpoint can also be downloaded from [[Google]](https://drive.google.com/drive/folders/1INhVq2XNcfyz8w2D71KP5Ucmhtmc9m01?usp=sharing).



### **5.4 Main Training**

```
python train.py --multigpu
```

Multi-GPU mode is required for contrastive learning.



Parameters can be set in the `./data/config.py` file. By default, we use the detection loss along with the following three losses

- `JIGSAW_33` for E(L) <=> H, 3x3 jigsaw self-supervised learning
- `CONTRASTIVE_TARGET` for E(L) up, contrastive learning on the target domain
- `TRANSFER_CONTRASTIVE` for H <=> D(H)

We also provide strategies that we have tried but finally not adopted

- `ROTATION` for rotation self-supervised learning
- `JIGSAW_22` for 2x2 jigsaw self-supervised learning
- `JIGSAW_41` for 4x1 jigsaw self-supervised learning
- `JIGSAW_14` for 1x4 jigsaw self-supervised learning
- `CONTRASTIVE` for contrastive learning on both source and target domain (shared head)
- `CONTRASTIVE_SOURCE` for contrastive learning on the source domain


### **5.5 Evaluation**

The checkpoint saved in training contains adaptation heads, which are by default not taken into consideration in [test.py](../test_code/test.py). Therefore, to evaluate trained models, please edit the [loading code](..//test_code/test.py#L208) into

```
net_param = torch.load('trainer_0.pth')
new_net_param = {}
for param_key in net_param:
    if param_key[:3] == 'net':
        new_net_param[param_key[4:]] = net_param[param_key]
net.load_state_dict(new_net_param)
```



------

This code is based on [DSFD](https://github.com/yxlijun/DSFD.pytorch) and [MoCo](https://github.com/facebookresearch/moco). Thanks for their great works!
