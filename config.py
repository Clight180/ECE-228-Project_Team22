import torch
import os

# File handling:
processedImsPath = './processed_images/'
rawImsPath = './raw_images/'
tensorDataPath = './TensorData/'
savedModelsPath = './savedModels/'
savedFigsPath = './saved_figs'


# Feature handling
modelNum = 000 # Set to pre-existing model to avoid training from epoch 1 , ow 000
datasetID = 000 # Set to pre-existing dataset to avoid generating a new one, ow 000
trainSize = 10
testSize = int(trainSize*.2)
imDims = 128
numAngles = 64
showSummary = False


# Hyperparameters:
num_epochs = 1
batchSize = 10
learningRate = 1e-3
weightDecay = 1e-5
AMSGRAD = True
LRS_Gamma = .95

# GPU acceleration:
USE_GPU = True
dtype = torch.float32

# Runtime vars
theta = None
dimFolder = '/imSize_{}/'.format(imDims)
anglesFolder = '/imDims_{}/'.format(imDims)
experimentFolder = '/Dataset_{}_Model_{}/'.format(datasetID,modelNum)
