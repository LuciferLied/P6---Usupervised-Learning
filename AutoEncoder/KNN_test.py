import numpy as np
from sklearn.model_selection import GridSearchCV
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.utils.data as data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import warnings
warnings.filterwarnings("ignore")


# set device
if torch.cuda.is_available():
    print('Using GPU')
    dtype = torch.float32
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

train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=False)

test_set = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

#Classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def KNN(train_data,train_labs,test_data,test_labs):

    print('train_data shape: ',train_data.shape)
    print('train_labs shape: ',train_labs.shape)
    print('test_data shape: ',test_data.shape)
    print('test_labs shape: ',test_labs.shape)
    
    KNN = KNeighborsClassifier()
    scaler = StandardScaler()
    
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.fit_transform(test_data)

    KNN.fit(train_data,train_labs)
    predicted = KNN.predict(test_data)


    #In order to pretty print output
    print('Accuracy is :',accuracy_score(predicted,test_labs)*100,'%')

    #List Hyperparameters to tune
    leaf_size = list(range(1,5))
    n_neighbors = list(range(1,8))
    p=[1,2]
    #convert to dictionary
    hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
    #Making model
    clf = GridSearchCV(KNN, hyperparameters, cv=10)
    best_model = clf.fit(train_data,train_labs)
    #Best Hyperparameters Value
    print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
    print('Best p:', best_model.best_estimator_.get_params()['p'])
    print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
    #Predict testing set
    y_pred = best_model.predict(test_data)
    #Check performance using accuracy
    print(accuracy_score(test_labs, y_pred))
    
    confmatrix = confusion_matrix(predicted,test_labs)

    plt.subplots(figsize=(6,6))
    sns.heatmap(confmatrix,annot=True,fmt=".1f",linewidths=1.5)
    plt.savefig('confusion_matrix.png')

def test_data():
    with torch.no_grad():
        train_data = torch.tensor([])
        train_labs = torch.tensor([])
        test_labs = torch.tensor([])
        test_data = torch.tensor([])

        for train_pics, train_labels in train_loader:
            train_pics = train_pics.to(device)
            train_codes, _ = model(train_pics)
            
            train_codes = train_codes.to('cpu')
            train_codes = torch.flatten(train_codes, 1)
            
            train_data = torch.cat((train_data, train_codes), 0)
            train_labs = torch.cat((train_labs, train_labels), 0)
            if len(train_data) > 5000:
                break
        
    with torch.no_grad():
        for test_pics, test_labels in test_loader:
            test_pics = test_pics.to(device)
            test_codes, _ = model(test_pics)
            
            test_codes = test_codes.to('cpu')
            test_codes = torch.flatten(test_codes, 1)
            
            test_data = torch.cat((test_data, test_codes), 0)
            test_labs = torch.cat((test_labs, test_labels), 0)
            if len(test_data) > 5000:
                break

        # Remove last element
        train_data = train_data[:-1]
        train_labs = train_labs[:-1]

        
    KNN(train_data, train_labs, test_data, test_labs)

test_data()