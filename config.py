import torch

# File/feature handling:
processedImsPath = './processed_images/'
rawImsPath = './raw_images/'
tensorDataPath = './TensorData/'
savedModelsPath = './savedModels/'
modelNum = 000 # Set to pre-existing model to avoid training from epoch 1 , ow 000
datasetID = 000 # Set to pre-existing dataset to avoid generating a new one, ow 000
trainSize = 100
testSize = int(trainSize*.2)
imDims = 128
numAngles = 12
showSummary = False


# Hyperparameters:
num_epochs = 5
batchSize = 10
learningRate = 1e-3
weightDecay = 1e-5
AMSGRAD = True
LRS_Gamma = .95

# GPU acceleration:
USE_GPU = True
dtype = torch.float32