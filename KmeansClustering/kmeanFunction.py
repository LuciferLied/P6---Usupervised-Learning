import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.cluster import MiniBatchKMeans
import time
import csv

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
        #if i==1:
            #print("\nINDEX\n")
            #print(index[:200],"\n\n")
            #print("CLUSTER LABEL VALUES: \n")
            #print(kmeansLabels[:200])

        #count all the train_targets where index == 1, and pick the biggest one
        #e.g. train_target == 0 corresponds to index 1 three times, train_target == 1 corresponds to index 1 eight times, etc.
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

def printPerformanceMetrics(number_labels, train_target_numpyArray):
    #Print and compute accuracy 
    #print("\nReference labels:")
    #print(reference_labels)

    #print("\nPredicted & actual values: ")
    #print(number_labels[:20].astype('int'))
    #print(train_target_numpyArray[:20],"\n")

    #Computers and prints accuracy
    accuracy = accuracy_score(number_labels, train_target_numpyArray)
    #print("\nAccuracy: ", round(accuracy*100), "%")
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
    accuracy = printPerformanceMetrics(number_labels, train_target_numpyArray)
    #func.printSpecificPicture(train_numpyArray, 0)
    time_elapsed = time.time()-start_time
    return time_elapsed, accuracy, kmeans

def runProgram(train_reshaped, train_target_numpyArray):
    iterations = int(input("Enter number of iterations as integer: "))
    total_clusters = int(input("Enter number of clusters as integer: "))

    i=iterations
    time_elapsed_list = []
    accuracy_list = []
    kmeans_list =[]
    print("\nInitializing computations\n")
    print("STATS: ")
    while i>0:
        time_elapsed, accuracy, kmeans = runClustering(train_reshaped, train_target_numpyArray, total_clusters)
        print("{:<2}:   Time elapsed: {:<5.3f}   |   Accuracy: {:<5.3f}".format(iterations-i, time_elapsed, accuracy))
        time_elapsed_list.append(time_elapsed)
        accuracy_list.append(accuracy)
        kmeans_list.append(kmeans)
        i-=1

    
    print("\nEnd computations")
    return accuracy_list, time_elapsed_list, kmeans_list

def statsPrint(accuracy_list, time_elapsed_list, model):
    print("\n\nAVERAGE ACCURACY: {:>6.3f}".format(sum(accuracy_list)/len(accuracy_list)))
    print("HIGHEST ACCURACY: {:>6.3f}".format(max(accuracy_list)))
    print("LOWEST ACCURACY: {:>7.3f}".format(min(accuracy_list)))
    print("________________________")
    print("\nAVERAGE TIME: {:>10.3f}".format(sum(time_elapsed_list)/len(time_elapsed_list)))
    print("SHORTEST TIME: {:>9.3f}".format(min(time_elapsed_list)))
    print("LONGEST TIME: {:>10.3f}\n\n".format(max(time_elapsed_list)))
    print("Number of clusters: {}".format(model[0].n_clusters))

    
def saveToCSV(accuracy_list, time_elapsed_list):
    stats_CSV = open('C:\Skole\projekt\P6---Usupervised-Learning\KmeansClustering\dataanalysis\CSVstats.csv', 'w')
    writer = csv.writer(stats_CSV)
    writer.writerow(time_elapsed_list)
    writer.writerow(accuracy_list)
