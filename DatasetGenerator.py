import torch
import numpy as np
import cv2
from skimage.transform import radon
from scipy.ndimage.interpolation import rotate
import numpy.matlib as nm
import time
from tqdm import tqdm
import ImagePreprocessing
import config
import os
import sys

class CT_Dataset(torch.utils.data.Dataset):
    def __init__(self,datasetID=000, dsize=10, saveDataset=False):
        '''
        folderDir points to directory with all truth images:

        folderDir/
            im1.jpg
            im2.jpg
            ...

        :param datasetID:
        :param dsize:
        :param saveDataset:
        '''

        ### store vars in obj ###
        self.processedImsPath = config.processedImsPath
        self.nSlices = config.numAngles
        self.imDims = config.imDims
        self.datasetID = datasetID
        self.datasetSize = dsize

        ### generate dataset if 000, else load ###
        if self.datasetID == 000:
            self.filesList = ImagePreprocessing.genData(self.datasetSize)

            time.sleep(.01)
            print('Creating slice projections.')
            time.sleep(.01)

            ### loop vars ###
            data = torch.empty((0,self.nSlices+1,self.imDims,self.imDims))
            config.theta = np.linspace(0., 180., self.nSlices, endpoint=False)
            folderDir = self.processedImsPath
            dimFolder = '/imDims_{}/'.format(config.imDims)

            ### create slice projections, stack data points in large tensor ###
            for idx in tqdm(range(len(self.filesList))):
                time.sleep(.01)

                ### read processed image ###
                fileId = self.filesList[idx]
                try:
                    tensorOut = cv2.imread(folderDir + dimFolder + fileId, cv2.COLOR_BGR2GRAY)
                    sinogram = radon(tensorOut, theta=config.theta, circle=False, preserve_range=True)
                except:
                    print('File not readable, idx: {}'.format(idx))
                    continue

                ### create nSlices number of slices of processed image ###
                sv_bp = []
                for i in range(self.nSlices):
                    sv = np.expand_dims(sinogram[:, i], 1)
                    sv_p = nm.repmat(sv, 1, sinogram.shape[0] * 2)
                    rotated = rotate(sv_p, angle=90 + config.theta[i], reshape=False)
                    rotated_cropped = self.center_crop(rotated, tensorOut.shape)
                    sv_bp.append(rotated_cropped[..., None])
                sv_bp = np.dstack(sv_bp)
                sv_bp = sv_bp.transpose(2, 0, 1)
                tensorOut = torch.unsqueeze(torch.tensor(tensorOut, dtype=config.dtype), dim=0)
                sv_bp_t = torch.tensor(sv_bp, dtype=config.dtype)
                tensorOut = torch.cat((tensorOut, sv_bp_t))

                ### stack datapoint to data tensor ###
                data = torch.cat((data,torch.unsqueeze(tensorOut,dim=0)))

            ### save tensor dataset ###
            datasetID = np.random.randint(100, 999)
            print('dataset_{}.pt complete. {} images processed'.format(datasetID, len(self.filesList)))
            self.data = data.type(config.dtype)
            if saveDataset:
                if not os.path.isdir(config.tensorDataPath + config.dimFolder): os.mkdir(config.tensorDataPath + config.dimFolder)
                filePath =  config.tensorDataPath + config.dimFolder + config.anglesFolder
                if not os.path.isdir(filePath): os.mkdir(filePath)
                torch.save(data, filePath + 'dataset_{}_size_{}.pt'.format(datasetID,config.trainSize))
                config.datasetID = datasetID
                config.experimentFolder = '/Dataset_{}_Model_{}/'.format(config.datasetID, config.modelNum)

        ### load pre-built tensor dataset ###
        else:
            filePath = config.tensorDataPath + config.dimFolder + config.anglesFolder
            data = torch.load(filePath + 'dataset_{}_size_{}.pt'.format(self.datasetID,config.trainSize))
            self.data = data.type(config.dtype)
            print('dataset_{}.pt loaded. {} images'.format(datasetID, len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Returns a tensor of shape [n,h,w]. n: number of slices + 1, (h,w) size of image
        :param idx:
        :return:
        '''
        return self.data[idx]

    def imgShape(self):
        return self[0][0].shape

    def center_crop(self, img, dim):
        """Returns center cropped image
        Args:
        img: image to be center cropped
        dim: dimensions (width, height) to be cropped
        """
        width, height = img.shape[1], img.shape[0]

        #### process crop width and height for max available dimension ###
        crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
        crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
        mid_x, mid_y = int(width / 2), int(height / 2)
        cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
        crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
        return crop_img

if __name__ == '__main__':
    '''
    sys.argv[1:] : Array of specifications for Dataset(s)
    numAngles imSize dSize numAngles imSize dSize ...
    ^______Dataset_______^ ^______Dataset_______^
    '''
    if len(sys.argv)>1:
        numParams = 3
        assert (len(sys.argv)-1)%numParams == 0, 'Incomplete dataset spec set!'
        args = sys.argv[1:]
        argsList = [int(arg) for arg in sys.argv[1:]]
        datasetList = list(range(1, len(argsList), numParams))
        for dataset, idx in enumerate(datasetList):
            specs = [argsList[i + dataset * numParams] for i in range(numParams)]
            print('Generating Dataset {} with {} num angles, {} imSize, {} dSize'.format(
                dataset + 1, specs[0], (specs[1],specs[1]), specs[2]))
            config.numAngles = specs[0]
            config.anglesFolder = '/nAngles_{}/'.format(config.numAngles)
            config.imDims = specs[1]
            config.dimFolder = '/imSize_{}/'.format(config.imDims)
            config.trainSize = specs[2]
            CT_Dataset(dsize=config.trainSize, saveDataset=True)
    else:
        raise AssertionError('Dataset must be generated with args: numAngles imSize dSize numAngles imSize dSize ...')