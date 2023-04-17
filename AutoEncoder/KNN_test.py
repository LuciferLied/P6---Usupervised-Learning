import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.utils.data as data
from util import utils as util
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from util import Models as Model
from sklearn.preprocessing import StandardScaler
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
model = torch.load('KNN.pth')
model.to(device)
model.eval()
print('model',model)


# Settings
batch_size = 256

# DataLoader
train_set = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# DataLoader
test_set = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

test_loader = data.DataLoader(test_set, batch_size=10000, shuffle=False)


def KNN(TextString,x_train,y_train,x_test,y_test):

    print('x_train shape: ',x_train.shape)
    print('x_test shape: ',x_test.shape)
    
    clf = KNeighborsClassifier(n_neighbors=2)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(x_train)
    scaled_test = scaler.fit_transform(x_test)

    clf.fit(scaled_train,y_train)
    y_pred = clf.predict(scaled_test)

    #In order to pretty print output
    print("Accuracy of {} KNN is %{}".format(TextString,accuracy_score(y_pred=y_pred,y_true=y_test)*100))

    confmatrix = confusion_matrix(y_pred=y_pred,y_true=y_test)

    plt.subplots(figsize=(6,6))
    sns.heatmap(confmatrix,annot=True,fmt=".1f",linewidths=1.5)
    plt.savefig('pics/{}.png'.format(TextString))

def test_data():
    with torch.no_grad():
        x_data = []
        x_data = torch.tensor(x_data)
        x_lab = []
        x_lab = torch.tensor(x_lab)

        for pics, labels in train_loader:
            pics = pics.to(device)
            x_train, decoded = model(pics)
            x_train = x_train.to('cpu')
            x_train = torch.flatten(x_train, 1)
            
            x_lab = torch.cat((x_lab, labels), 0)
            x_data = torch.cat((x_data, x_train), 0)
        
    with torch.no_grad():
        for pics, labels in test_loader:
            pics = pics.to(device)
            codes, decoded = model(pics)
            codes = codes.to('cpu')
            codes = torch.flatten(codes, 1)
            
        x_train = np.array(x_data)
        x_train = x_train[:-1]
        y_train = np.array(x_lab)
        y_train = y_train[:-1]
        x_test = np.array(codes)
        y_test = np.array(labels)
        
    KNN("CIFAR10",x_train, y_train, x_test, y_test)
