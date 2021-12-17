import cv2
import numpy as np
import glob

image_paths = sorted(glob.glob('/home/mark/Downloads/data_road/data_road/training/image_2/*.png'))
label_paths = sorted(glob.glob('/home/mark/Downloads/data_road/data_road/training/gt_image_2/*.png'))

for image, label in zip(image_paths, label_paths):
    img = cv2.imread(image)
    label_img = cv2.imread(label, 0)
    blank = np.zeros((375, 1242, 3), dtype=np.uint8)
    img[label_img == 105] = (255, 255, 255)
    # img[blank] = (255, 255, 255)
    cv2.imshow('img', img)
    cv2.waitKey(0)
