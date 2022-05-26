import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import model
import Dataset
from torch.utils.data import DataLoader
from torchsummary import summary
import time
from tqdm import tqdm

start = time.time()


##### CONFIGURATIONS #####


# File/feature handling:
processedImsPath = './processed_images/'
rawImsPath = './raw_images/'
tensorDataPath = './TensorData/'
ContinueLearning = False
modelNum = 000 # Set to pre-existing model to avoid training from epoch 1 , ow 000
datasetID = 431 #Set to pre-existing dataset to avoid generating a new one, ow 000
numIms = 250
imDims = 160
numAngles = 16

# Hyperparameters:
num_epochs = 10
batchSize = 10
learningRate = 1e-3
weightDecay = 1e-5
AMSGRAD = True
LRS_Gamma = .95

# GPU acceleration:
USE_GPU = True
dtype = torch.float32


##### PRE-TRAINING SETUP #####


if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)

# Constructing data handlers:
data = Dataset.CT_Dataset(processedImsPath, imDims, numAngles, datasetID=datasetID, datasetSize=numIms)
data_DL = DataLoader(data, batch_size=batchSize)
print('dataset_{}.pt loaded. Size of dataset: {}'.format(datasetID, len(data)))

print('Time of dataset completion: {:.2f}'.format(time.time()-start))

# Constructing NN:
myNN = model.DBP_NN(channelsIn=numAngles, filtSize=imDims)
summary(myNN)
print('Model generated. Model ID: {}'.format(myNN.modelId))

if ContinueLearning:
    myNN.load_state_dict(torch.load('{}/NN_StateDict_{}.pt'.format(processedImsPath, modelNum)))
    print('Loaded model: {}'.format(modelNum))

myNN.to(device)
myNN.train()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(myNN.parameters(),lr=learningRate,weight_decay=weightDecay,amsgrad=AMSGRAD)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LRS_Gamma)

trainLoss = []
bestModel = myNN.state_dict()
bestLoss = 10e10


##### TRAINING ROUTINE #####


# Training routine:
for epoch in range(num_epochs):
    Loss = 0
    numBatches = 0

    time.sleep(.02)

    for idx, im in enumerate(tqdm(data_DL)):
        batchLoss = 0
        numBatches+=1
        optimizer.zero_grad()
        targetIm = im[:,0,:,:].cuda()
        input = im[:, 1:, :, :].cuda()
        if device.type == 'cuda':
            out = myNN(input)
            batchLoss = criterion(torch.squeeze(out,dim=1),targetIm)
        else:
            raise NotImplementedError
        Loss += float(batchLoss)
        batchLoss.backward()
        optimizer.step()

    trainLoss.append(Loss/numBatches)

    if trainLoss[-1] < bestLoss:
        bestLoss = trainLoss[-1]
        bestModel = myNN.state_dict()

    scheduler.step()
    print('{}/{} epochs completed. Loss: {}'.format(epoch+1,num_epochs,float(trainLoss[-1])))

print('done')
print('Time at training completion: {:.2f}'.format(time.time()-start))


##### POST-TRAINING ROUTINE #####

myNN.load_state_dict(bestModel)
config.modelNum = myNN.modelId
torch.save(bestModel,'{}/NN_StateDict_{}.pt'.format('./savedModels/',myNN.modelId))

plt.figure()
plt.plot(trainLoss)
plt.title('Model ID: {}\nBatch Size: {}, Initial Learning Rate: {},\n LRS_Gamma: {}, amsgrad: {}, weight decay: {}'.format(myNN.modelId,batchSize,learningRate,LRS_Gamma,AMSGRAD,weightDecay))
plt.show()

myNN.eval()
for i in np.random.randint(0,len(data)-1,5):
    im = data[i]
    testOrig = im[0,:,:]

    testOut = myNN(torch.unsqueeze(im[1:,:,:].cuda(),0))
    testOut = torch.squeeze(testOut)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(testOrig)
    plt.title('Original\nIm Size: {}'.format((imDims,imDims)))
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(testOut.cpu().detach().numpy())
    plt.title('DCNN Output\nNum Angles: {}, Best Loss: {:.2e}'.format(numAngles, bestLoss), y=-.2)
    plt.axis('off')
    plt.suptitle('Dataset Size: {}, Model ID: {}'.format(len(data),myNN.modelId))
    plt.show()

print('Time to completion: {:.2f}'.format(time.time()-start))
exit('Training Complete. Dataset num: {}, Model num: {}'.format(config.datasetID,config.modelNum))

