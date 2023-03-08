import kmeanFunction as func
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2



class CNN:
    def __init__(self):
        self.con = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
        
class Kernel():
    def __init__(self, kernelxaxis, kernelyaxis):
        self.kernel = np.empty(shape=(kernelxaxis, kernelyaxis))

x = 3
y = 3

k = Kernel(x, y)
cnn = CNN()

k.kernel[1][1] = 1
k.kernel[0][1] = 1
k.kernel[1][0] = 1
k.kernel[2][1] = 1
k.kernel[1][2] = 1

print(k.kernel)
train_data, test_data = func.downloadData(datasets.MNIST)

#INITIALIZE DATA CHANGE LATER

train_numpyList = []
test_numpyList = []

train_purified, train_target = zip(*train_data)
test_purified, test_target = zip(*test_data)
train_target_numpyArray = np.array(train_target)
x = 0
for x in train_purified:
    train_numpyList.append(x.numpy())

for x in test_purified:
    test_numpyList.append(x.numpy())

train_numpyArray=np.array(train_numpyList)

train_numpyArray= train_numpyArray.squeeze()
print("train_numpyArray shape: ", train_numpyArray.shape)
print(train_numpyArray)

#FINISH INITIALIZING DATA

func.printSpecificPicture(train_numpyArray, 0)
#cv2.filter2D(train_numpyArray[0],1,k.kernel)

print(cnn.con(train_data[0][0]))
