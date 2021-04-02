#### [Noise Synthesis H_noise] Training Code for HLA-Face: Joint High-Low Adaptation for Low Light Face Detection (CVPR21)

The official PyTorch implementation **(partial training code)** for HLA-Face: Joint High-Low Adaptation for Low Light Face Detection (CVPR21).

You can find more information on our [project website](https://daooshee.github.io/HLA-Face-Website/).

------



##### 1. Requirements

- Python 3
- opencv



##### 2. Blurring E(L) and generate E(L)_blur

```
python 'Blurring_E(L).py'
```

This code will automatically blur the images in `../dataset/DarkFace/images_enhanced/`, and save the results in `../dataset/DarkFace/images_enhanced_denoise/`.



##### 3. Training Pix2Pix to learn the mapping from E(L)_blur to E(L) for noise synthesis

We use the [Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) project.

First use the `datasets/combine_A_and_B.py` script in Pix2Pix to organize the data

```
python ./datasets/combine_A_and_B.py --fold_A 'YOUR_PATH/train_code/dataset/DarkFace/images_enhanced_denoise/' --fold_B 'YOUR_PATH/train_code/dataset/DarkFace/images_enhanced/' --fold_AB ./datasets/DARKFACE_pix2pix_train
```

Then train Pix2Pix

```
python train.py --dataroot ./datasets/DARKFACE_pix2pix_train --name DARKFACE_noise_synthesis_pix2pix --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 --preprocess crop --batch_size 8
```

If you want to skip the training process, the trained models can be downloaded from [[Google]](https://drive.google.com/file/d/1AmTirjBe765vyRK1u4N4wzwc4rpvklo6/view?usp=sharing), [[Baidu (u9cj)]](https://pan.baidu.com/s/1p2vYlqGxfwV7u6yfWGQ_LA).



##### 4. Blurring H and generate H_blur

```
python Blurring_H.py
```

This script will automatically blur the images in `../dataset/WiderFace/WIDER_train/images/`, and save the results in `../dataset/WiderFace/WIDER_train_denoise/`.

This script also pads the image, so that the resolution can be divided by 256 (to support Pix2Pix).



##### 5. Applying the learned Pix2Pix on H_blur and generate H_noise

Link the dataset

```
ln -s YOUR_PATH/train_code/dataset/WiderFace/WIDER_train_denoise ./datasets/WIDERFACE_pix2pix_test/testA
ln -s YOUR_PATH/train_code/dataset/WiderFace/WIDER_train_denoise ./datasets/WIDERFACE_pix2pix_test/testB
```

Apply the learned Pix2Pix

```
python test.py --dataroot ./datasets/WIDERFACE_pix2pix_test --name DARKFACE_noise_synthesis_pix2pix --model pix2pix --netG unet_256 --dataset_mode unaligned --norm batch --preprocess reisze --num_test 1000000
```



Finally crop and reorganize the images

```
python Reorganize_H_noise.py
```

This script will crop and organize the Pix2Pix results from `PATH_TO_Pix2Pix/results/DARKFACE_noise_synthesis_pix2pix/test_latest/images/` to `../dataset/WiderFace/WIDER_images_add_noise/train/`.

Please edit the `PATH_TO_Pix2Pix` variable in `Reorganize_H_noise.py` if your path to Pix2Pix is not in `./PixPix`.