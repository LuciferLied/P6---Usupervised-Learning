import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.utils.data as data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from util import utils as util
from util import Models as Model
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import time

torch.manual_seed(0)

# set device
if torch.cuda.is_available():
    print('Using GPU')
    device = torch.device('cuda')
else:
    print('Using CPU')
    device = torch.device('cpu')

# Load model
model = torch.load('Cif10.pth')
model.to(device)

# Settings
batch_size = 256

# DataLoader
train_set = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_set = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
knn_train_loader = data.DataLoader(train_set, batch_size=5000, shuffle=False)

test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
knn_test_loader = data.DataLoader(test_set, batch_size=5000, shuffle=False)

images, labels = next(iter(train_loader))
print(images.shape)

def KNN(train_data,train_labs,test_data,test_labs):
    # Log time
    start_time = time.time()
    
    KNN = KNeighborsClassifier(n_neighbors=200)
    
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.fit_transform(test_data)
    
    
    KNN.fit(train_data,train_labs)
    predicted = KNN.predict(test_data)

    #In order to pretty print output
    print('KNN Accuracy',accuracy_score(predicted,test_labs)*100,'%')

    # #List Hyperparameters to tune
    # leaf_size = list(range(1,2))
    # n_neighbors = list(range(1,8))
    # p=[1,2]
    # #convert to dictionary
    # hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
    # #Making model
    # clf = GridSearchCV(KNN, hyperparameters, cv=10)
    # best_model = clf.fit(train_data,train_labs)
    # #Best Hyperparameters Value
    # print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
    # print('Best p:', best_model.best_estimator_.get_params()['p'])
    # print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
    # #Predict testing set
    # y_pred = best_model.predict(test_data)
    # #Check performance using accuracy
    # print(accuracy_score(test_labs, y_pred))
    
    print("--- %s seconds ---" % (time.time() - start_time))
    # # confmatrix = confusion_matrix(predicted,test_labs)

    # plt.subplots(figsize=(6,6))
    # sns.heatmap(confmatrix,annot=True,fmt=".1f",linewidths=1.5)
    # plt.savefig('confusion_matrix.png')

def test_data():

    train_imgs, train_labs = next(iter(knn_train_loader))
    test_imgs , test_labs  = next(iter(knn_test_loader))
    
    train_codes, _ = model(train_imgs.to(device))
    test_codes, _  = model(test_imgs.to(device))
    
    KNN(train_codes.flatten(1).detach().cpu().numpy(), train_labs, test_codes.flatten(1).detach().cpu().numpy(), test_labs)

test_data()