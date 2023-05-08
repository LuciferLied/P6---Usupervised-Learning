import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV



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

def test_best_knn(KNN,train_data,train_labs,test_data,test_labs,neighbors):
    start_time = time.time()
    
    print('Testing best KNN')
    # List Hyperparameters to tune
    leaf_size = list(range(1, 2))
    n_neighbors = list(range(1, neighbors))
    p = [1, 2]
    # convert to dictionary
    hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
    # Making model
    clf = GridSearchCV(KNN, hyperparameters, cv=10)
    best_model = clf.fit(train_data, train_labs)
    # Best Hyperparameters Value
    print('Best leaf_size:',
          best_model.best_estimator_.get_params()['leaf_size'])
    print('Best p:', best_model.best_estimator_.get_params()['p'])
    print('Best n_neighbors:',
          best_model.best_estimator_.get_params()['n_neighbors'])
    # Predict testing set
    y_pred = best_model.predict(test_data)
    # Check performance using accuracy
    print(accuracy_score(test_labs, y_pred))
    print("--- %s seconds ---" % (time.time() - start_time))
    