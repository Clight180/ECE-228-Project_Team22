import torch

# File handling:
processedImsPath = './processed_images/'
rawImsPath = './raw_images/'
tensorDataPath = './TensorData/'
savedModelsPath = './savedModels/'
savedFigsPath = './saved_figs'


# Feature handling
modelNum = 777 # Set to pre-existing model to avoid training from epoch 1 , ow 000
datasetID = 771 # Set to pre-existing dataset to avoid generating a new one, ow 000
trainSize = 1000
testSize = int(trainSize*.2)
imDims = 128
numAngles = 64
showSummary = False


# Hyperparameters:
num_epochs = 15
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
