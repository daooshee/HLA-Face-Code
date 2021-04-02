import cv2
import os
import glob
import random

def showNimages(name, label_path, image_path, GT=False):
    image = cv2.imread(image_path)

    with open(label_path) as f:
        txt_data = f.readlines()
    if GT:
        for data in txt_data[1:]:
            print(data, len(data))
            x, y, w, h = data.split(' ')[:4]
            x, y, w, h = float(x), float(y), float(w), float(h)
            x1, y1, x2, y2 = int(x), int(y), int(w), int(h)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255))
            cv2.rectangle(image, (x1-1, y1-1), (x2+1, y2+1), (0, 255, 255))
            cv2.rectangle(image, (x1+1, y1+1), (x2-1, y2-1), (0, 255, 255))
    else:
        for data in txt_data:
            x, y, w, h, cf = data.split(' ')[:5]
            x, y, w, h, cf = float(x), float(y), float(w), float(h), float(cf)
            x1, y1, x2, y2 = int(x), int(y), int(w), int(h)
            if cf > 0.6:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, int(255*cf)))
                cv2.rectangle(image, (x1-1, y1-1), (x2+1, y2+1), (0, 255, int(255*cf)))
                cv2.rectangle(image, (x1+1, y1+1), (x2-1, y2-1), (0, 255, int(255*cf)))
        
    cv2.imwrite('vis_{}.png'.format(name), image)

showNimages('test', 'result/test.txt', 'test.png')
