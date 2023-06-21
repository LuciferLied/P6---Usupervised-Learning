from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import statistics
from util import CSV as csvUtil

files = 'baseline30.csv', 'simCLR2/Cifar10-conv_big.csv', 'simCLR2/Cifar10-conv_small.csv', 'simCLR2/Cifar10-linear_big.csv', 'simCLR2/Cifar10-linear_small.csv', 'simCLR2/Cifar10-mobile.csv', 'simCLR2/Cifar10-squeezenet.csv'

#figure, axis = plt.subplots(2, 2, figsize=(20, 20))
plt.figure(figsize=(5,5))

for file in files:
    print(file)
    data = csvUtil.readFromCSV(file)
    data_holder = []

    i = 1
    for index in data:
            i = i + 1
            data[data.index(index)] = [float(index[0]), float(index[1]), float(index[2]), float(index[3]), float(index[4]), float(index[5])]

    epochs = []
    batch_size = []
    learning_rate = []
    loss = []
    knn_acc = []
    kmeans_acc = []

    for i in data:
            epochs.append(i[0]) #"Epochs"
            batch_size.append(i[1]) #"Batch Size"
            learning_rate.append(i[2]) #"Learning Rate"
            loss.append(i[3]) #"Loss"
            knn_acc.append(i[4]) #"Accuracy"
            kmeans_acc.append(i[5]) #"Accuracy 2"      
            
    ep_sep = []
    batch_size_sep = []
    lr_sep = []
    mean_acc_batch = []
    mean_time_batch = []
    highest_acc_ep_knn = []
    highest_acc_ep_kmeans = []
    mean_acc_ep_knn = []
    mean_acc_ep_kmeans = []
    
    for i in range(len(epochs)):
        if epochs[i] not in ep_sep:
            ep_sep.append(epochs[i])
        if batch_size[i] not in batch_size_sep:
            batch_size_sep.append(batch_size[i])
        if learning_rate[i] not in lr_sep:
            lr_sep.append(learning_rate[i])

    ep_sep = sorted(ep_sep)
    batch_size_sep = sorted(batch_size_sep)
    lr_sep = sorted(lr_sep)
    
    for i in range(len(ep_sep)):
        holder = []
        for j in range(len(data)):
            if data[j][0] == ep_sep[i]:
                holder.append(data[j][4])
        highest_acc_ep_knn.append(max(holder))
        mean_acc_ep_knn.append(statistics.mean(holder))
        

    for i in range(len(ep_sep)):
        holder = []
        for j in range(len(data)):
            if data[j][0] == ep_sep[i]:
                holder.append(data[j][5])
        highest_acc_ep_kmeans.append(max(holder))
        mean_acc_ep_kmeans.append(statistics.mean(holder))
        
    plt.plot(ep_sep, highest_acc_ep_knn)
    #plt.plot(ep_sep, highest_acc_ep_kmeans)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Baseline', 'Conv Big', 'Conv Small', 'Linear Big', 'Linear Small', 'MobileNet', 'SqueezeNet'], loc='upper left')
    
    
"""     axis[0,0].plot(ep_sep, highest_acc_ep_knn)
    #axis[0,0].plot(ep_sep, highest_acc_ep_kmeans)
    axis[0,0].set_xlabel('Epochs')
    axis[0,0].set_ylabel('Accuracy')
    axis[0,0].legend(['Baseline', 'Conv Big', 'Conv Small', 'Linear Big', 'Linear Small', 'MobileNet', 'SqueezeNet'], loc='upper left') """

plt.tight_layout()
plt.savefig('graphs/SimCLR.png')
plt.show()