import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
# import model
# import datasetGenerator as DG
from torch.utils.data import DataLoader

# File/feature handling:
filePath = 'data'
numAngles = 16
imDim = 64
ContinueLearning = False
modelNum = 706
numInputs = 1000


# Hyperparameters:
num_epochs = 25
batchSize = 10
learningRate = 1e-2
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
imDims = (imDim,imDim)
Dataset1 = myData = DG.Dataset(list(np.linspace(1,1,numInputs).astype(int)),'{}/scans'.format(filePath),'{}/sv_bp'.format(filePath),'.jpg','.jpg', numAngles,imDim)
BP_DataLoader = DataLoader(Dataset1, batch_size=batchSize)
print('Dataloader initialized. Number of inputs: {}'.format(numInputs))

# Constructing NN:
myNN = model.DBP_NN(channelsIn=numAngles)
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
    batchLoss = 0
    numBatches = 0
    for idx, im in enumerate(BP_DataLoader):
        optimizer.zero_grad()
        out = myNN(im[0][:,:,:,:].cuda())
        loss = criterion(out,im[1][:,:,:,:].cuda())
        batchLoss += float(loss)
        numBatches +=1
        loss.backward()
        optimizer.step()
    trainLoss.append(batchLoss/numBatches)
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
for i in np.random.randint(0,numInputs-1,5):
    im = Dataset1[i]
    testOrig = im[1][0,:,:]

    testOut = myNN(torch.unsqueeze(torch.tensor(im[0][:,:,:]),0).cuda())
    testOut = torch.squeeze(testOut)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(testOrig)
    plt.title('Original\nIm Size: {}'.format((imDims,imDims)))
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(testOut.cpu().detach().numpy())
    plt.title('DCNN Output\nNum Angles: {}, Best Loss: {:.2f}'.format(numAngles, bestLoss), y=-.2)
    plt.axis('off')
    plt.suptitle('Dataset Size: {}, Model ID: {}'.format(len(Dataset1),myNN.modelId))
    plt.show()




