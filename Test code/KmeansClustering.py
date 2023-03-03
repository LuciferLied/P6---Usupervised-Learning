import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.cluster import MiniBatchKMeans
import timeit
import kmeanFunction as func


warnings.filterwarnings("ignore")
print("RUN")
train_data, test_data = func.downloadData(datasets.MNIST)
train_reshaped, train_target_numpyArray, test_reshaped, test_target_numpyArray, train_numpyArray, test_numpyArray = func.purifyData(train_data, test_data)
kmeansLabels = func.generateClusters(train_reshaped, test_target_numpyArray)
reference_labels = func.retrieveInfo(kmeansLabels,train_target_numpyArray)
number_labels = func.assignPredictions(kmeansLabels, reference_labels)
func.printPerformanceMetrics(reference_labels, number_labels, train_target_numpyArray)
func.printSpecificPicture(train_numpyArray, 0)

print("FINISHED")