import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import model
import DatasetGenerator
from torch.utils.data import DataLoader
from torchsummary import summary
import time
from tqdm import tqdm
import config
from skimage.transform import radon, iradon
import os
import cv2

def Experiment():
    start = time.time()

    ##### PRE-TRAINING SETUP #####

    if config.USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')

    print('using device:', device)

    ### Constructing data handlers ###
    print('Loading training data.')
    train_dataset = DatasetGenerator.CT_Dataset(datasetID=config.datasetID,dsize=config.trainSize,saveDataset=True)
    print('Loading testing data.')
    test_dataset = DatasetGenerator.CT_Dataset(dsize=config.testSize)
    train_DL = DataLoader(train_dataset, batch_size=config.batchSize)
    test_DL = DataLoader(test_dataset, batch_size=config.batchSize)

    print('Time of dataset completion: {:.2f}'.format(time.time()-start))

    ### Constructing NN ###
    myNN = model.DBP_NN(channelsIn=config.numAngles, filtSize=config.imDims)
    if config.modelNum != 000:
        myNN.load_state_dict(torch.load('{}/NN_StateDict_{}.pt'.format(config.savedModelsPath, config.modelNum)))
        myNN.modelId = config.modelNum
        print('Loaded model num: {}'.format(myNN.modelId))
    else:
        config.modelNum = myNN.modelId
        print('Model generated. Model ID: {}'.format(myNN.modelId))

    if config.showSummary:
        summary(myNN)
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
        for iter, im in enumerate(tqdm(train_DL)):
            trainBatchLoss = 0
            optimizer.zero_grad()
            targetIm = im[:,0,:,:].cuda()
            input = im[:, 1:, :, :].cuda()
            if device.type == 'cuda':
                out = myNN(input.type(config.dtype))
                trainBatchLoss = criterion(torch.squeeze(out, dim=1), targetIm)
            else:
                raise NotImplementedError
            trainEpochLoss += float(trainBatchLoss)
            trainBatchLoss.backward()
            optimizer.step()
            targetIm.detach()
            input.detach()
        scheduler.step()
        trainLoss.append(trainEpochLoss / len(train_DL))

        ### validation batch testing ###
        myNN.eval()
        with torch.no_grad():
            valEpochLoss = 0
            for iter, im in enumerate(test_DL):
                targetIm = im[:, 0, :, :].cuda()
                input = im[:, 1:, :, :].cuda()
                if device.type == 'cuda':
                    out = myNN(input)
                    valBatchLoss = criterion(torch.squeeze(out, dim=1), targetIm)
                else:
                    raise NotImplementedError
                valEpochLoss += float(valBatchLoss)
                targetIm.detach()
                input.detach()
            validationLoss.append(valEpochLoss / len(test_DL))

        ### store best model ###
        if trainLoss[-1] < bestLoss:
            bestLoss = trainLoss[-1]
            bestModel = myNN.state_dict()

        print('{}/{} epochs completed. Train loss: {:.4f}, validation loss: {:.4f}'.format(epoch+1,config.num_epochs,
                                                                                   float(trainLoss[-1]),
                                                                                   float(validationLoss[-1])))

    print('done')
    print('Time at training completion: {:.2f}'.format(time.time()-start))


    ##### POST-TRAINING ROUTINE #####

    myNN.load_state_dict(bestModel)
    config.modelNum = myNN.modelId
    torch.save(bestModel,'{}/NN_StateDict_{}.pt'.format('./savedModels/',myNN.modelId))


    ### Figure saving ###

    savedFigsPath = config.savedFigsPath
    dimFolder = config.dimFolder
    anglesFolder = config.anglesFolder
    experimentFolder = config.experimentFolder
    if not os.path.isdir(savedFigsPath + dimFolder): os.mkdir(savedFigsPath + dimFolder)
    if not os.path.isdir(savedFigsPath + dimFolder + anglesFolder): os.mkdir(savedFigsPath + dimFolder + anglesFolder)
    filePath = savedFigsPath + dimFolder + anglesFolder + experimentFolder
    if not os.path.isdir(filePath): os.mkdir(filePath)



    ### Observing Results ###

    plt.figure()
    plt.plot(trainLoss, label='Train Loss')
    plt.plot(validationLoss, label='Validation Loss')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.title('Model ID: {}, Dataset ID {}\nBatch Size: {}, Initial Learning Rate: {},\n '
              'LRS_Gamma: {}, amsgrad: {}, weight decay: {}'.format(myNN.modelId,config.datasetID,config.batchSize,
                                                                    config.learningRate,config.LRS_Gamma,
                                                                    config.AMSGRAD,config.weightDecay))
    plt.savefig(filePath + 'LossPlot.jpg')
    if config.printFigs:
        plt.show()

    myNN.eval()
    for iter, i in enumerate(np.random.randint(0, len(train_dataset) - 1, 3)):
        im = train_dataset[i]
        testOrig = im[0,:,:]
        testOut_unsqueezed = myNN(torch.unsqueeze(im[1:,:,:].cuda(),0))
        testOut = torch.squeeze(testOut_unsqueezed)

        sinogram = radon(testOrig.numpy(), theta=config.theta, circle=False, preserve_range=True)
        FBP_Out = iradon(sinogram, theta=config.theta,circle=False,preserve_range=True)


        cv2.imwrite(filePath + 'Original_{}.jpg'.format(i + 1),testOrig.numpy())
        cv2.imwrite(filePath + 'DCNN_{}.jpg'.format(iter + 1), testOut.cpu().detach().numpy())
        cv2.imwrite(filePath + 'FBP_{}.jpg'.format(iter + 1), FBP_Out)


        plt.figure()

        plt.subplot(2,2,1)
        plt.imshow(testOrig)
        plt.title('Original')
        plt.axis('off')

        plt.subplot(2,2,2)
        plt.imshow(testOut.cpu().detach().numpy())
        plt.title('DCNN')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(FBP_Out)
        plt.title('Filtered Back Projection', y=-.2)
        plt.axis('off')

        plt.subplot(2,2,4)
        plt.text(.2,.3,'Dataset Size: {}\nDataset ID: {}\nModel ID: {}\n Im Size: {}\nNum Angles: {}\nBest Loss: {:.2e}'.format(config.trainSize,config.datasetID,config.modelNum,(config.imDims,config.imDims),config.numAngles,bestLoss))
        plt.axis('off')
        plt.savefig(filePath + 'Subplot_{}.jpg'.format(iter + 1))
        if config.printFigs:
            plt.show()


    print('Time to completion: {:.2f}'.format(time.time()-start))
    print('Training Complete. Dataset num: {}, Model num: {}'.format(config.datasetID,config.modelNum))


if __name__ == '__main__':
    '''
    sys.argv[1:] : Array of nAngles intended
    Run Experiments by number of angles
    '''
    config.printFigs = False
    if sys.argv:
        args = sys.argv[1:]
        nAnglesList = sys.argv[1:]
        for nAngles in nAnglesList:
            print('Running Experiment with {} num angles'.format(nAngles))
            config.numAngles = int(nAngles)
            config.anglesFolder = '/nAngles_{}/'.format(config.numAngles)
            config.modelNum = 000
            config.datasetID = 000
            Experiment()
    else:
        Experiment()