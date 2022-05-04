import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import model
import Dataset
from torch.utils.data import DataLoader

# File/feature handling:
filePath = './data/data'
ContinueLearning = False
modelNum = 000

# Hyperparameters:
num_epochs = 25
batchSize = 10
learningRate = 5e-2
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
data = Dataset.CT_Dataset(filePath)
imDims = data.imgShape()[0]
numAngles = data[0].shape[0]-1
data_DL = DataLoader(data, batch_size=batchSize)
print('Dataloader initialized. Size of dataset: {}'.format(len(data)))

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
    Loss = 0
    for idx, im in enumerate(data_DL):
        optimizer.zero_grad()
        if device.type == 'cuda':
            out = myNN(im[:,1:,:,:].cuda())
            batchLoss = criterion(out,im[:,1:,:,:].cuda())
        else:
            out = myNN(im[:, 1:, :, :])
            batchLoss = criterion(out, im[:, 0, :, :])
        Loss += float(batchLoss)
        batchLoss.backward()
        optimizer.step()
    trainLoss.append(Loss/len(data))
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
    plt.title('DCNN Output\nNum Angles: {}, Best Loss: {:.2f}'.format(numAngles, bestLoss), y=-.2)
    plt.axis('off')
    plt.suptitle('Dataset Size: {}, Model ID: {}'.format(len(data),myNN.modelId))
    plt.show()

