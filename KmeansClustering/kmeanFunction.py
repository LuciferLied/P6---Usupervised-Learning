import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.cluster import MiniBatchKMeans
import time
import csv
import os

def downloadData(dataset):
    # Download training data from open datasets.
    train_data = dataset(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = dataset(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    return train_data, test_data

def purifyData(train_data, test_data):
    train_purified, train_target = zip(*train_data)
    test_purified, test_target = zip(*test_data)


    train_numpyList = []
    test_numpyList = []

    x = 0
    for x in train_purified:
        train_numpyList.append(x.numpy())

    for x in test_purified:
        test_numpyList.append(x.numpy())


    #convert targets to NP_arrays
    train_target_numpyArray = np.array(train_target)
    #test_target_numpyArray = np.array(test_target)

    train_numpyArray=np.array(train_numpyList)
    train_numpyArray= train_numpyArray.squeeze()

    test_numpyArray=np.array(test_numpyList)
    test_numpyArray= test_numpyArray.squeeze()

    train_reshaped=train_numpyArray.reshape(len(train_numpyArray),-1)
    #test_reshaped=test_numpyArray.reshape(len(test_numpyArray),-1)
    
    return train_reshaped, train_target_numpyArray

#Generates clusters based on pixel values
def generateClusters(train_reshaped, total_clusters):
    #total_clusters = len(np.unique(test_target_numpyArray))
    kmeans=MiniBatchKMeans(n_clusters = total_clusters)
    kmeans.fit(train_reshaped)
    kmeansLabels = kmeans.labels_
    #print("\nKmeans Labels: ")
    #print(kmeans.labels_[:20])        
    return kmeansLabels, kmeans

#Associates each cluster with most probable label
def retrieveInfo(kmeansLabels,train_target_numpyArray):
    reference_labels={}
    for i in range(len(np.unique(kmeansLabels))):
        #If cluster label = i, assign 1. Otherwise assign 0.
        index = np.where(kmeansLabels == i,1,0)
        num = np.bincount(train_target_numpyArray[index==1]).argmax()
        reference_labels[i]=num
    return reference_labels
    
def assignPredictions(kmeansLabels, reference_labels):
    #Initializes the array - ignore random.rand
    number_labels = np.random.rand(len(kmeansLabels))

    #For picture 0, find corresponding kmeans_label. Then find the actual number that kmeans.label points to.
    for i in range(len(kmeansLabels)):
        number_labels[i]=reference_labels[kmeansLabels[i]]

    return number_labels

def computeAccuracy(number_labels, train_target_numpyArray):
    accuracy = accuracy_score(number_labels, train_target_numpyArray)
    return accuracy

    #Prints picture
def printSpecificPicture(array, index):
    plt.gray()
    plt.imshow(array[index])
    plt.show()


def runClustering(train_reshaped, train_target_numpyArray, total_clusters):
    
    start_time = time.time()
    kmeansLabels, kmeans = generateClusters(train_reshaped, total_clusters)
    reference_labels = retrieveInfo(kmeansLabels,train_target_numpyArray)
    number_labels = assignPredictions(kmeansLabels, reference_labels)
    accuracy = computeAccuracy(number_labels, train_target_numpyArray)
    #func.printSpecificPicture(train_numpyArray, 0)   ADD train_numpyArray back to call
    time_elapsed = time.time()-start_time
    return time_elapsed, accuracy, kmeans

def runProgram(train_reshaped, train_target_numpyArray):
    iterations = int(input("Enter number of iterations as integer: "))
    total_clusters = int(input("Enter number of clusters as integer: "))

    i=iterations
    time_elapsed_list = []
    accuracy_list = []
    inertia_list = []
    homogeneity_list = []
    print("\nInitializing computations\n")
    print("STATS: ")
    while i>0:
        time_elapsed, accuracy, kmeans = runClustering(train_reshaped, train_target_numpyArray, total_clusters)
        print("{:<2}:   Time elapsed: {:<5.3f}   |   Accuracy: {:<5.3f}   |   Inertia: {:<5.3f}   |   Homogeneity: {:<5.3f}".format(iterations-i, time_elapsed, accuracy, kmeans.inertia_, metrics.homogeneity_score(train_target_numpyArray, kmeans.labels_)))
        time_elapsed_list.append(time_elapsed)
        accuracy_list.append(accuracy)
        inertia_list.append(kmeans.inertia_)
        homogeneity_list.append(metrics.homogeneity_score(train_target_numpyArray, kmeans.labels_))
        i-=1

    
    print("\nEnd computations")
    return accuracy_list, time_elapsed_list, inertia_list, homogeneity_list, total_clusters

def statsPrint(accuracy_list, time_elapsed_list, inertia_list, homogeneity_list, total_clusters):
    print("\n\nAVERAGE ACCURACY: {:>8.3f}".format(sum(accuracy_list)/len(accuracy_list)))
    print("HIGHEST ACCURACY: {:>8.3f}".format(max(accuracy_list)))
    print("LOWEST ACCURACY: {:>9.3f}".format(min(accuracy_list)))
    
    print("________________________")

    print("\nAVERAGE INERTIA: {:>15.1f}".format(sum(inertia_list)/len(inertia_list)))
    print("HIGHEST INERTIA: {:>15.1f}".format(max(inertia_list)))
    print("LOWEST INERTIA: {:>16.1f}".format(min(inertia_list)))

    print("________________________")

    print("\nAVERAGE HOMOGENEITY: {:>5.3f}".format(sum(homogeneity_list)/len(homogeneity_list)))
    print("HIGHEST HOMOGENEITY: {:>5.3f}".format(max(homogeneity_list)))
    print("LOWEST HOMOGENEITY: {:>6.3f}".format(min(homogeneity_list)))

    print("________________________")

    print("\nAVERAGE TIME: {:>12.3f}".format(sum(time_elapsed_list)/len(time_elapsed_list)))
    print("SHORTEST TIME: {:>11.3f}".format(min(time_elapsed_list)))
    print("LONGEST TIME: {:>12.3f}\n\n".format(max(time_elapsed_list)))
    print("Number of clusters: {:>3}".format(total_clusters))

def saveToCSV(accuracy_list, time_elapsed_list):

    rootdir = os.getcwd()

    for subdir, dirs, files in os.walk(rootdir):
        
        filepath = subdir + os.sep

        if filepath.__contains__("dataanalysis"):
            filepath = filepath + "CSVstats.csv"
            with open(filepath, 'w', newline='') as CSVstats:
                writer = csv.writer(CSVstats)
                writer.writerow(time_elapsed_list)
                writer.writerow(accuracy_list)