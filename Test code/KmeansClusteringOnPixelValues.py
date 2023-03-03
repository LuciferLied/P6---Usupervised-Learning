import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.cluster import MiniBatchKMeans



# Download training data from open datasets.
train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

print("RUN")

train_purified, train_target = zip(*train_data)
test_purified, test_target = zip(*test_data)


train_numpyList = []
train_target_numpyList = []
test_numpyList = []
test_taget_numpyList = []

x = 0
for x in train_purified:

    train_numpyList.append(x.numpy())

for x in test_purified:

    test_numpyList.append(x.numpy())


#convert targets to NP_arrays
train_target_numpyArray = np.array(train_target)
test_target_numpyArray = np.array(test_target)



train_numpyArray=np.array(train_numpyList)
train_numpyArray= train_numpyArray.squeeze()

test_numpyArray=np.array(test_numpyList)
test_numpyArray= test_numpyArray.squeeze()

train_reshaped=train_numpyArray.reshape(len(train_numpyArray),-1)
test_reshaped=test_numpyArray.reshape(len(test_numpyArray),-1)

#Generates clusters
total_clusters = len(np.unique(test_target_numpyArray))
kmeans=MiniBatchKMeans(n_clusters = total_clusters)
kmeans.fit(train_reshaped)
kmeans.labels_
print("\nKmeans Labels: ")
print(kmeans.labels_[:20])

#Associates each cluster with most probable label
def retrieve_info(cluster_labels,train_target_numpyArray):
    reference_labels={}
    for i in range(len(np.unique(kmeans.labels_))):
        index = np.where(cluster_labels == i,1,0)
        num = np.bincount(train_target_numpyArray[index==1]).argmax()
        reference_labels[i]=num
    return reference_labels 

#Assign labels
reference_labels = retrieve_info(kmeans.labels_,train_target_numpyArray)
number_labels = np.random.rand(len(kmeans.labels_))
for i in range(len(kmeans.labels_)):
    number_labels[i]=reference_labels[kmeans.labels_[i]]

#Print and compute accuracy
print("\nReference labels:")
print(reference_labels)

print("\nPredicted & actual values: ")
print(number_labels[:20].astype('int'))
print(train_target_numpyArray[:20],"\n\n")

accuracy_score = (accuracy_score(number_labels, train_target_numpyArray))
print("\nAccuracy: ", round(accuracy_score*100), "%")

#Prints picture
plt.gray()
plt.imshow(train_numpyArray[0])
plt.show()

print("FINISHED")


