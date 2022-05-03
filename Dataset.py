import os
import torchvision.io
from torchvision.io import read_image
import torch

class CT_Dataset(torch.utils.data.Dataset):
    def __init__(self, dir):
        '''
        ./data/data is assumed to have file heirarchy:

        file_no1 /
            trueIm.jpg
            back_projections_file_no1 /
                file_no1_0.jpg
                file_no1_1.jpg
                ...
        file_no2 /
        ...

        Use: >> data = Dataset.CT_Dataset('./data/data')
             >> data[0]
             Out[8]:
             tensor([[2, 2, 2,  ..., 2, 2, 2],
                     [2, 2, 2,  ..., 2, 2, 2],
                     [2, 2, 2,  ..., 2, 2, 2],
                     ...,
                     [2, 2, 2,  ..., 2, 2, 2],
                     [2, 2, 2,  ..., 2, 2, 2],
                     [2, 2, 2,  ..., 2, 2, 2]], dtype=torch.uint8)

             >> len(data): returns number of data points (1st level subfolders in ./data/data)

             >> data.imgShape(): returns a list of int image shape

        :param dir: dir is the location where data point subfolders are located
        '''
        self.dir = dir
        self.dp_table = list(enumerate(list(os.walk(self.dir))[0][1]))
        self.filetype = '.jpg'
        self.trainImDir = '/back_projections_'

    def __len__(self):
        return len(self.dp_table)

    def __getitem__(self, idx):
        '''
        Returns a tensor of shape [n,h,w]. n: number of slices + 1, (h,w) size of image
        :param idx:
        :return:
        '''
        folderId = self.dp_table[idx][1]
        folderDir = self.dir + '/' + folderId
        tensorOut = read_image(folderDir + '/' + folderId + self.filetype, mode=torchvision.io.ImageReadMode.GRAY)
        trainIm_table = list(enumerate(list(os.walk(folderDir + self.trainImDir + folderId))[0][2]))
        for i in range(len(trainIm_table)):
            trainImSlice = read_image(folderDir + self.trainImDir + folderId + '/' + trainIm_table[i][1], mode=torchvision.io.ImageReadMode.GRAY)
            tensorOut = torch.cat((tensorOut, trainImSlice))
        return tensorOut

    def imgShape(self):
        return self[0][0].shape
