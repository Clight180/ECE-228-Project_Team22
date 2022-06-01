import torch

# File handling:
processedImsPath = './processed_images/'
rawImsPath = './raw_images/'
tensorDataPath = './TensorData/'
savedModelsPath = './savedModels/'
savedFigsPath = './saved_figs'


# Feature handling
numAngles = 64
imDims = 128
trainSize = 1000
testSize = int(trainSize*.2)
modelNum = 000 # Set to pre-existing model to avoid training from epoch 1 , ow 000
datasetID = 000 # Set to pre-existing dataset to avoid generating a new one, ow 000
showSummary = False
printFigs = True


# Hyperparameters:
num_epochs = 35
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
anglesFolder = '/nAngles_{}/'.format(numAngles)
experimentFolder = '/Dataset_{}_Model_{}/'.format(datasetID,modelNum)
