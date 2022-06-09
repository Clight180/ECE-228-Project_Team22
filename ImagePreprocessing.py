import os
import numpy as np
import cv2
from tqdm import tqdm
import random
import time
import config

def genData(dsize=128):
    '''
    Generates data from raw images. Does preprocessing, saves in folders and returns data.
    :param img_size:
    :param sizeData:
    :return:
    '''

    ### create dim folder if not exists ###
    dimFolder = '/imDims_{}/'.format(config.imDims)
    if not os.path.isdir(config.processedImsPath + dimFolder): os.mkdir(config.processedImsPath + dimFolder)

    ### load and shuffle from original dataset ###
    files = os.listdir(config.rawImsPath)
    random.shuffle(files)
    datasetFiles = files[0:dsize]

    time.sleep(.01)
    print('Preprocessing Images.')
    time.sleep(.01)

    ### preprocess images ###
    for fileID in tqdm(datasetFiles):
        time.sleep(.01)

        ### point to image, if exists skip ###
        filePath = config.processedImsPath + dimFolder + fileID
        if os.path.exists(filePath):
            continue

        ### load image as grayscale ###
        image = cv2.imread(config.rawImsPath + fileID)
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ### preprocessing ###
        grayImage = cv2.resize(grayImage, [config.imDims, config.imDims])
        grayImage[int(config.imDims * 5 / 6):] = 0
        kernel = np.ones((5, 5), np.uint8)
        kernel2 = np.ones((3, 3), np.uint8)
        grayImage = cv2.erode(grayImage, kernel, iterations=1)
        grayImage = cv2.dilate(grayImage, kernel2, iterations=1)
        grayImage = cv2.erode(grayImage, kernel2, iterations=1)
        grayImage = cv2.dilate(grayImage, kernel2, iterations=1)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        grayImage = clahe.apply(grayImage)

        ### save to folder, return list of files ###
        cv2.imwrite(config.processedImsPath + dimFolder + fileID, grayImage)
    return datasetFiles