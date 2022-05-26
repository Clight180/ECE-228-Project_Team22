import os
import numpy as np
import cv2
from tqdm import tqdm
import random
import time

data_set_path = './processed_images/'
if not os.path.isdir(data_set_path): os.mkdir(data_set_path)

imgset_path = './raw_images/'
files = os.listdir(imgset_path)
random.shuffle(files)

def genData(img_size, sizeData):
    datasetFiles = files[0:sizeData]

    time.sleep(.1)
    print('Preprocessing Images.')
    time.sleep(.01)

    for item in tqdm(datasetFiles):
        time.sleep(.01)

        if os.path.exists(data_set_path+item):
            continue

        path = imgset_path + '\\' + item

        ### resize and apply filter ###
        image = cv2.imread(path)
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = cv2.resize(grayImage, [img_size, img_size])
        grayImage[int(img_size * 5 / 6):] = 0

        kernel = np.ones((5, 5), np.uint8)
        kernel2 = np.ones((3, 3), np.uint8)
        grayImage = cv2.erode(grayImage, kernel, iterations=1)
        grayImage = cv2.dilate(grayImage, kernel2, iterations=1)
        grayImage = cv2.erode(grayImage, kernel2, iterations=1)
        grayImage = cv2.dilate(grayImage, kernel2, iterations=1)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        grayImage = clahe.apply(grayImage)

        cv2.imwrite(data_set_path + item, grayImage)
    return datasetFiles