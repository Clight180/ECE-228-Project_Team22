import os
from torchvision.io import read_image
import torch
import torchvision.io


imDims = 160
numAngles = 24

filePath = './New_folder/newData/'
dpTable = list(enumerate(list(os.walk(filePath))[0][1]))
filetype = '.jpg'
trainImDir = '/back_projections_'

for idx in range(len(dpTable)):
    folderId = dpTable[idx][1]
    folderDir = filePath + '/' + folderId
    tensorOut = read_image(folderDir + '/' + folderId + filetype, mode=torchvision.io.ImageReadMode.GRAY)
    try:
        trainIm_table = list(enumerate(list(os.walk(folderDir + trainImDir + folderId))[0][2]))
    except:
        print('Image not correct. folder: ' + str(dpTable[idx]))
    for i in range(len(trainIm_table)):
        trainImSlice = read_image(folderDir + trainImDir + folderId + '/' + trainIm_table[i][1],
                                  mode=torchvision.io.ImageReadMode.GRAY)
        try:
            tensorOut = torch.cat((tensorOut, trainImSlice))
        except:
            print('imdims off')
    if tensorOut.shape[0] != numAngles + 1 or tensorOut.shape[1] != imDims or tensorOut.shape[2] != imDims:
        print('Image not correct. folder: ' + str(dpTable[idx]))
