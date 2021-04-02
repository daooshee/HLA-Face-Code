import os
import glob
import cv2

PATH_TO_Pix2Pix = './Pix2Pix'


if not os.path.exists("../dataset/WiderFace/WIDER_images_add_noise/train/"):
	os.makedirs("../dataset/WiderFace/WIDER_images_add_noise/train/")

PathList = glob.glob("../dataset/WiderFace/WIDER_train/images/*/*.*")

for img_path in PathList:
	img_name = os.path.basename(img_path)
	new_img_path = f'{PATH_TO_Pix2Pix}/results/DARKFACE_noise_synthesis_pix2pix/test_latest/images/' + img_name.split('.')[0]+'_fake_B.png'

	if not os.path.exists(new_img_path):
		continue
	else:
		print(new_img_path)

	img = cv2.imread(img_path)
	H,W,C = img.shape
	new_img = cv2.imread(new_img_path)
	new_img = new_img[256:256+H, 256:256+W, :]

	cv2.imwrite("../dataset/WiderFace/WIDER_images_add_noise/train/" + img_name, new_img)