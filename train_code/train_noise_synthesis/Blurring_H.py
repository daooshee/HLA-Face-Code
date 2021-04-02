import os
import glob
import cv2
import numpy as np

def gasuss_noise(image, mean, var):
    noise = np.random.normal(mean, var, [image.shape[0], image.shape[1], 1])
    out = image + np.tile(noise,3)
    return np.clip(out, 0, 255).astype(np.uint8)

def reshape(img, border=256):
    H, W, C = img.shape

    new_H = H + border*2
    if new_H % border != 0:
        new_H += border - new_H % border

    new_W = W + border*2
    if new_W % border != 0:
        new_W += border - new_W % border

    new_img = cv2.copyMakeBorder(img, border, new_H-border-H,
                                      border, new_W-border-W, cv2.BORDER_REFLECT)
    return new_img

save_folder = '../dataset/WiderFace/WIDER_train_denoise/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

for img_path in glob.glob("../dataset/WiderFace/WIDER_train/images/*/*.*"):
    image_name = os.path.basename(img_path)
    target_path = save_folder + image_name
    print(target_path)
    
    img = cv2.imread(img_path)
    img = gasuss_noise(img, 0, 16)
    blurred = cv2.bilateralFilter(img, 25, 75, 75)
    blurred = cv2.medianBlur(blurred, 5)

    padded_blurred = reshape(blurred)
    
    cv2.imwrite(target_path, padded_blurred)