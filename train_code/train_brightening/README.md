## [Brightening  E(L)] Training Code for HLA-Face: Joint High-Low Adaptation for Low Light Face Detection (CVPR21)

The official PyTorch implementation **(partial training code)** for HLA-Face: Joint High-Low Adaptation for Low Light Face Detection (CVPR21).

You can find more information on our [project website](https://daooshee.github.io/HLA-Face-Website/).

------



### 1. Requirements

- Python 3
- PyTorch (I use version 1.60. I think other versions would also be OK)
- opencv
- torchvision



### 2. Data preparation

Download the training dataset from https://github.com/Li-Chongyi/Zero-DCE, and put it in `./data/train_data`.



### 3. Train

```
python lowlight_train.py
```



### 4. Test

Rename the checkpoint as `./Illumination-Enhancer.pth`, and run:

```
python lowlight_test.py 
```

This script will enhance all images in `../dataset/DarkFace/images/train/`.

The results will be saved in `../dataset/DarkFace/images_enhanced/train/`.



------

This code is based on [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE). Thanks a lot for the great work!