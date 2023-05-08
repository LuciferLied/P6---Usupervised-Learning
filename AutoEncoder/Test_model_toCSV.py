from matplotlib import pyplot as plt
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
from util import utils
import seaborn as sns


def KNN(train_data, train_labs, test_data, test_labs, neighbors):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    KNN = KNeighborsClassifier(n_neighbors=neighbors)

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.fit_transform(test_data)

    KNN.fit(train_data, train_labs)
    predicted = KNN.predict(test_data)
    
    accuracy = accuracy_score(predicted, test_labs)*100
    # In order to pretty print output
    print('KNN Accuracy', accuracy, '%')
    
    # utils.test_best_knn(KNN,train_data,train_labs,test_data,test_labs)
    
    # confmatrix = confusion_matrix(predicted, test_labs)
    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.xaxis.tick_top()
    # sns.heatmap(confmatrix, annot=True, fmt=".1f", linewidths=1.5, annot_kws={'size': 8}, xticklabels=classes, yticklabels=classes)
    # plt.yticks(rotation=0)
    # plt.savefig('pics/confusion_matrix.png')
    return accuracy

def kmeans(train_data, train_labs, clusters):
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    kmeans = KMeans(n_clusters=clusters, n_init='auto').fit(train_data)
    
    labels = train_labs
    # cast to int
    labels = np.array(labels)
    labels = labels.astype(int)
    
    labels = np.array(labels)
    ref_labels = utils.retrieveInfo(kmeans.labels_, labels)
    num_predicted = utils.assignPredictions(kmeans.labels_, ref_labels)
    accuracy = utils.computeAccuracy(num_predicted, labels)

    accuracy = round(accuracy * 100,2)
    print('Kmeans Accuracy', accuracy, '%')
    return accuracy

def test(model, train_loader, test_loader, device, neighbors_cluster):
    with torch.no_grad():
        train_codes = torch.tensor([])
        train_labs = torch.tensor([])
        test_codes = torch.tensor([])
        test_labs = torch.tensor([])
        
        print('loading test...')
        
        for pics, labels in train_loader:
            codes, _ = model(pics.to(device))
            train_codes = torch.cat((train_codes, codes.flatten(1).cpu()), 0)
            train_labs = torch.cat((train_labs, labels), 0)
            if len(train_codes) > 9000:
                break

        for pics,labels in test_loader:
            codes, _ = model(pics.to(device))
            test_codes = torch.cat((test_codes, codes.flatten(1).cpu()), 0)
            test_labs = torch.cat((test_labs, labels), 0)
            if len(test_codes) > 9000:
                break
        
        print('test loaded')
        print('testing...')
    knn_acc = KNN(train_codes, train_labs, test_codes, test_labs,neighbors_cluster)
    kmeans_acc = kmeans(train_codes,train_labs,neighbors_cluster)
    
    return knn_acc, kmeans_acc