import numpy as np
from sklearn.cluster import KMeans
import torch
from torchsummary import summary
from torchvision.transforms import ToTensor
import torch.utils.data as data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torchvision import datasets, transforms
from util import utils
import time

# set device
if torch.cuda.is_available():
    print('Using GPU')
    device = torch.device('cuda')
else:
    print('Using CPU')
    device = torch.device('cpu')

# Load model
model = torch.load('trained_models/Res181.pth')
model.to(device)
model.eval()

# Settings
batch_size = 256

# DataLoader
train_set = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_set = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

def KNN(train_data, train_labs, test_data, test_labs):
    # Log time

    KNN = KNeighborsClassifier(n_neighbors=150)

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.fit_transform(test_data)

    KNN.fit(train_data, train_labs)
    predicted = KNN.predict(test_data)

    # In order to pretty print output
    print('KNN Accuracy', accuracy_score(predicted, test_labs)*100, '%')
    # utils.test_best_knn(KNN,train_data,train_labs,test_data,test_labs)
    
    confmatrix = confusion_matrix(predicted, test_labs)
    plt.subplots(figsize=(6, 6))
    sns.heatmap(confmatrix, annot=True, fmt=".1f", linewidths=1.5)
    plt.savefig('pics/confusion_matrix.png')


def kmeans(train_codes,train_labs):
    kmeans = KMeans(n_clusters=50, n_init='auto').fit(train_codes)
    
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


def test_knn():
    with torch.no_grad():

        train_codes = torch.tensor([])
        train_labs = torch.tensor([])
        test_codes = torch.tensor([])
        test_labs = torch.tensor([])
        
        for pics,labels in train_loader:
            codes, _ = model(pics.to(device))
            train_codes = torch.cat((train_codes, codes.flatten(1).cpu()), 0)
            train_labs = torch.cat((train_labs, labels), 0)
            if len(train_codes) > 5000:
                break

        for pics,labels in test_loader:
            codes, _ = model(pics.to(device))
            test_codes = torch.cat((test_codes, codes.flatten(1).cpu()), 0)
            test_labs = torch.cat((test_labs, labels), 0)
            if len(test_codes) > 5000:
                break
        
        print('train_codes',train_codes.shape)
        print('test_codes',test_codes.shape)
    KNN(train_codes, train_labs,test_codes, test_labs)
    kmeans(train_codes,train_labs)
        
test_knn()

