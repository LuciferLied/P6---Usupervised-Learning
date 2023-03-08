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


class CNN:
    def __init__(x):
        x = 1

class Kernel:
    def __init__(x):
        originalPictures = x

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

print("train_numpyArray shape: ", train_numpyArray.shape)
print(train_numpyArray)

#FINISH INITIALIZING DATA