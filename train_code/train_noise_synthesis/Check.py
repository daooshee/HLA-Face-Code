import os
import glob
import random

# CUT, CycleGAN is OK

NAME = 'MUNIT'

os.system('rm -r check_paired_data')
os.mkdir('check_paired_data')

img_list = glob.glob('/mnt/hdd/wangwenjing/Dataset/WiderFace/original/WIDER_val/images/*/*.*')
for i in range(10):
	wider_img = random.choice(img_list)
	new_img = wider_img.replace('/original/', '/{}/'.format(NAME))
	if os.path.exists(new_img):
		os.system('cp {} check_paired_data/{}_val_ori_img.jpg'.format(wider_img,i))
		os.system('cp {} check_paired_data/{}_val_new_img.jpg'.format(new_img,i))
	else:
		print(new_img, 'not exsits')

img_list = glob.glob('/mnt/hdd/wangwenjing/Dataset/WiderFace/original/WIDER_train/images/*/*.*')
for i in range(10):
	wider_img = random.choice(img_list)
	new_img = wider_img.replace('/original/', '/{}/'.format(NAME))
	if os.path.exists(new_img):
		os.system('cp {} check_paired_data/{}_train_ori_img.jpg'.format(wider_img,i))
		os.system('cp {} check_paired_data/{}_train_new_img.jpg'.format(new_img,i))
	else:
		print(new_img, 'not exsits')
