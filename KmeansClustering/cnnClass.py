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


train_purified, train_target = zip(*train_data)
test_purified, test_target = zip(*test_data)

print("train_data type: ", type(train_data))
print("train_purified: ", type(train_purified))
print(train_purified[0])