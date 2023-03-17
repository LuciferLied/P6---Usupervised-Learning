import numpy as np
from sklearn.metrics import accuracy_score


def computeAccuracy(num_predicted, train_target_numpyArray):
    accuracy = accuracy_score(num_predicted, train_target_numpyArray)
    return accuracy

def assignPredictions(kmeansLabels, reference_labels):
    #Initializes the array - ignore random.rand
    num_predicted = np.random.rand(len(kmeansLabels))

    #For picture 0, find corresponding kmeans_label. Then find the actual number that kmeans.label points to.
    for i in range(len(kmeansLabels)):
        num_predicted[i]=reference_labels[kmeansLabels[i]]

    return num_predicted

#Associates each cluster with most probable label
def retrieveInfo(kmeansLabels, train_target_numpyArray):
    reference_labels={}
    for i in range(len(np.unique(kmeansLabels))):
        #If cluster label = i, assign 1. Otherwise assign 0.
        index = np.where(kmeansLabels == i,1,0)
        num = np.bincount(train_target_numpyArray[index==1]).argmax()
        reference_labels[i]=num
    return reference_labels