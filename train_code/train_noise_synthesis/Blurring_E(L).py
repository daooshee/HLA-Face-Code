import cv2
import glob
import os

if not os.path.exists("../dataset/DarkFace/images_enhanced_denoise/train/"):
	os.makedirs("../dataset/DarkFace/images_enhanced_denoise/train/")

for img_path in glob.glob("../dataset/DarkFace/images_enhanced/train/*.*"):
	print(img_path, img_path.replace('images_enhanced','images_enhanced_denoise'))
	img = cv2.imread(img_path)
	blurred = cv2.bilateralFilter(img, 25, 75, 75)
	blurred = cv2.medianBlur(blurred, 5)
	cv2.imwrite(img_path.replace('images_enhanced','images_enhanced_denoise'), blurred)