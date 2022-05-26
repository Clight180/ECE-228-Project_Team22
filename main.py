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
import config

start = time.time()

##### PRE-TRAINING SETUP #####


if config.USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)

### Constructing data handlers ###
print('Loading training data.')
train_data = Dataset.CT_Dataset(datasetID=config.datasetID,dsize=config.trainSize,saveDataset=True)
print('Loading testing data.')
test_data = Dataset.CT_Dataset(dsize=config.testSize)
train_DL = DataLoader(train_data, batch_size=config.batchSize)
test_DL = DataLoader(test_data, batch_size=config.batchSize)

print('Time of dataset completion: {:.2f}'.format(time.time()-start))

### Constructing NN ###
myNN = model.DBP_NN(channelsIn=config.numAngles, filtSize=config.imDims)
if config.showSummary:
    summary(myNN)
print('Model generated. Model ID: {}'.format(myNN.modelId))
if config.ContinueLearning:
    myNN.load_state_dict(torch.load('{}/NN_StateDict_{}.pt'.format(config.processedImsPath, config.modelNum)))
    print('Loaded model: {}'.format(config.modelNum))
myNN.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(myNN.parameters(),lr=config.learningRate,
                             weight_decay=config.weightDecay,amsgrad=config.AMSGRAD)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.LRS_Gamma)

### training metrics ###
trainLoss = []
validationLoss = []
bestModel = myNN.state_dict()
bestLoss = 10e10

##### TRAINING ROUTINE #####

### training routine ###
for epoch in range(config.num_epochs):
    time.sleep(.01)

    myNN.train()
    trainEpochLoss = 0
    ### train batch training ###
    for idx, im in enumerate(tqdm(train_DL)):
        trainBatchLoss = 0
        optimizer.zero_grad()
        targetIm = im[:,0,:,:].cuda()
        input = im[:, 1:, :, :].cuda()
        if device.type == 'cuda':
            out = myNN(input)
            trainBatchLoss = criterion(torch.squeeze(out, dim=1), targetIm)
        else:
            raise NotImplementedError
        trainEpochLoss += float(trainBatchLoss)
        trainBatchLoss.backward()
        optimizer.step()
    scheduler.step()
    trainLoss.append(trainEpochLoss / len(train_DL))

    ### validation batch testing ###
    myNN.eval()
    with torch.no_grad():
        valEpochLoss = 0
        for idx, im in enumerate(test_DL):
            targetIm = im[:, 0, :, :].cuda()
            input = im[:, 1:, :, :].cuda()
            if device.type == 'cuda':
                out = myNN(input)
                valBatchLoss = criterion(torch.squeeze(out, dim=1), targetIm)
            else:
                raise NotImplementedError
            valEpochLoss += float(valBatchLoss)
        validationLoss.append(valEpochLoss / len(test_DL))

    ### store best model ###
    if trainLoss[-1] < bestLoss:
        bestLoss = trainLoss[-1]
        bestModel = myNN.state_dict()

    print('{}/{} epochs completed. Train loss: {}, validation loss: {}'.format(epoch+1,config.num_epochs,
                                                                               float(trainLoss[-1]),
                                                                               float(validationLoss[-1])))

print('done')
print('Time at training completion: {:.2f}'.format(time.time()-start))


##### POST-TRAINING ROUTINE #####

myNN.load_state_dict(bestModel)
config.modelNum = myNN.modelId
torch.save(bestModel,'{}/NN_StateDict_{}.pt'.format('./savedModels/',myNN.modelId))

plt.figure()
plt.plot(trainLoss, label='Train Loss')
plt.plot(validationLoss, label='Validation Loss')
plt.yscale('log')
plt.legend(loc='upper right')
plt.title('Model ID: {}\nBatch Size: {}, Initial Learning Rate: {},\n '
          'LRS_Gamma: {}, amsgrad: {}, weight decay: {}'.format(myNN.modelId,config.batchSize,
                                                                config.learningRate,config.LRS_Gamma,
                                                                config.AMSGRAD,config.weightDecay))
plt.show()

myNN.eval()
for i in np.random.randint(0, len(train_data) - 1, 5):
    im = train_data[i]
    testOrig = im[0,:,:]

    testOut = myNN(torch.unsqueeze(im[1:,:,:].cuda(),0))
    testOut = torch.squeeze(testOut)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(testOrig)
    plt.title('Original\nIm Size: {}'.format((config.imDims,config.imDims)))
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(testOut.cpu().detach().numpy())
    plt.title('DCNN Output\nNum Angles: {}, Best Loss: {:.2e}'.format(config.numAngles, bestLoss), y=-.2)
    plt.axis('off')
    plt.suptitle('Dataset Size: {}, Model ID: {}'.format(len(train_data), myNN.modelId))
    plt.show()

print('Time to completion: {:.2f}'.format(time.time()-start))
exit('Training Complete. Dataset num: {}, Model num: {}'.format(config.datasetID,config.modelNum))

