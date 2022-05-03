import torch.nn as nn
import numpy as np

filt1,filt2 = 64, 64
kernelSize = (3,3)
Padding = 1

class DBP_NN(nn.Module):
    def __init__(self,channelsIn):
        super(DBP_NN,self).__init__()

        self.modelId = np.random.randint(100,999)

        self.c1 = nn.Sequential(
            nn.Conv2d(channelsIn, filt1, kernelSize, padding=Padding),
            nn.ReLU()
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(filt1, filt1, kernelSize, padding=Padding),
            nn.BatchNorm2d(filt1),
            nn.ReLU()
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(filt1, filt1, kernelSize, padding=Padding),
            nn.BatchNorm2d(filt1),
            nn.ReLU()
        )
        self.c4 = nn.Sequential(
            nn.Conv2d(filt1, filt1, kernelSize, padding=Padding),
            nn.BatchNorm2d(filt1),
            nn.ReLU()
        )
        self.c5 = nn.Sequential(
            nn.Conv2d(filt1, filt1, kernelSize, padding=Padding),
            nn.BatchNorm2d(filt1),
            nn.ReLU()
        )
        self.c6 = nn.Sequential(
            nn.Conv2d(filt1, filt1, kernelSize, padding=Padding),
            nn.BatchNorm2d(filt1),
            nn.ReLU()
        )
        self.c7 = nn.Sequential(
            nn.Conv2d(filt1, filt1, kernelSize, padding=Padding),
            nn.BatchNorm2d(filt1),
            nn.ReLU()
        )
        self.c8 = nn.Sequential(
            nn.Conv2d(filt1, filt1, kernelSize, padding=Padding),
            nn.BatchNorm2d(filt1),
            nn.ReLU()
        )
        self.c9 = nn.Sequential(
            nn.Conv2d(filt1, filt1, kernelSize, padding=Padding),
            nn.BatchNorm2d(filt1),
            nn.ReLU()
        )
        self.c10 = nn.Sequential(
            nn.Conv2d(filt1, filt1, kernelSize, padding=Padding),
            nn.BatchNorm2d(filt1),
            nn.ReLU()
        )
        self.c11 = nn.Sequential(
            nn.Conv2d(filt1, filt2, kernelSize, padding=Padding),
            nn.BatchNorm2d(filt2),
            nn.ReLU()
        )
        self.c12 = nn.Sequential(
            nn.Conv2d(filt2, filt2, kernelSize, padding=Padding),
            nn.BatchNorm2d(filt2),
            nn.ReLU()
        )
        self.c13 = nn.Sequential(
            nn.Conv2d(filt2, filt2, kernelSize, padding=Padding),
            nn.BatchNorm2d(filt2),
            nn.ReLU()
        )
        self.c14 = nn.Sequential(
            nn.Conv2d(filt2, filt2, kernelSize, padding=Padding),
            nn.BatchNorm2d(filt2),
            nn.ReLU()
        )
        self.c15 = nn.Sequential(
            nn.Conv2d(filt2, filt2, kernelSize, padding=Padding),
            nn.BatchNorm2d(filt2),
            nn.ReLU()
        )
        self.c16 = nn.Sequential(
            nn.Conv2d(filt2, filt2, kernelSize, padding=Padding),
            nn.BatchNorm2d(filt2),
            nn.ReLU()
        )

        self.f1 = nn.Sequential(
            nn.Conv2d(filt2, 1, kernelSize, padding=Padding)
        )

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x4 = self.c4(x3)
        x5 = self.c5(x4)
        x6 = self.c6(x5)
        x7 = self.c7(x6)
        x8 = self.c8(x7)
        x9 = self.c9(x8)
        x10 = self.c10(x9)
        x11 = self.c11(x10)
        x12 = self.c12(x11)
        x13 = self.c13(x12)
        x14 = self.c14(x13)
        x15 = self.c15(x14)
        x16 = self.c16(x15)
        x17 = self.f1(x16)
        return x17