import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import model
import Dataset
from torch.utils.data import DataLoader
from torchsummary import summary

# File/feature handling:
filePath = './New_folder/newData/'
ContinueLearning = False
modelNum = 000
imDims = 160
numAngles = 24
verifyDat = False

# Hyperparameters:
num_epochs = 50
batchSize = 10
learningRate = 1e-3
weightDecay = 1e-5
AMSGRAD = True
LRS_Gamma = .912

# GPU acceleration:
USE_GPU = True
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)

# Constructing data handlers:
data = Dataset.CT_Dataset(filePath, imDims, numAngles, verifyDat)
data_DL = DataLoader(data, batch_size=batchSize)
print('Dataloader initialized. Size of dataset: {}'.format(len(data)))

# Constructing NN:
myNN = model.DBP_NN(channelsIn=numAngles)
summary(myNN)
print('Model being generated. Model ID: {}'.format(myNN.modelId))


if ContinueLearning:
    myNN.load_state_dict(torch.load('{}/NN_StateDict_{}.pt'.format(filePath,modelNum)))
    print('Loaded model: {}'.format(modelNum))

myNN.to(device)
myNN.train()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(myNN.parameters(),lr=learningRate,weight_decay=weightDecay,amsgrad=AMSGRAD)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LRS_Gamma)

trainLoss = []
bestModel = myNN.state_dict()
bestLoss = 10e10

# Training routine:
for epoch in range(num_epochs):
    Loss = 0
    numBatches = 0
    for idx, im in enumerate(data_DL):
        batchLoss = 0
        numBatches+=1
        optimizer.zero_grad()
        if device.type == 'cuda':
            im = im[:, 1:, :, :].cuda()
            out = myNN(im)
            batchLoss = criterion(out,im[:,1:,:,:].cuda())
        else:
            im = torch.nn.functional.normalize(im[:, 1:, :, :])
            out = myNN(im)
            batchLoss = criterion(out, im[:, 0, :, :])
        Loss += float(batchLoss)
        batchLoss.backward()
        optimizer.step()
        del im
        del batchLoss
    trainLoss.append(Loss/numBatches)
    if trainLoss[-1] < bestLoss:
        bestLoss = trainLoss[-1]
        bestModel = myNN.state_dict()

    scheduler.step()
    print('{}/{} epochs completed. Loss: {}'.format(epoch+1,num_epochs,float(trainLoss[-1])))

print('done')

myNN.load_state_dict(bestModel)
torch.save(bestModel,'{}/NN_StateDict_{}.pt'.format(filePath,myNN.modelId))

plt.figure()
plt.plot(trainLoss)
plt.title('Model ID: {}\nBatch Size: {}, Initial Learning Rate: {},\n LRS_Gamma: {}, amsgrad: {}, weight decay: {}'.format(myNN.modelId,batchSize,learningRate,LRS_Gamma,AMSGRAD,weightDecay))
plt.show()

myNN.eval()
for i in np.random.randint(0,len(data)-1,5):
    im = data[i]
    testOrig = im[0,:,:]

    testOut = myNN(torch.unsqueeze(torch.tensor(im[1:,:,:]),0).cuda())
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

