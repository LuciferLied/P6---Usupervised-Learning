import torch
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
model = torch.load('trained_models/Cif10_70.pth')
model.to(device)

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
knn_train_loader = data.DataLoader(train_set, batch_size=5000, shuffle=False)

test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
knn_test_loader = data.DataLoader(test_set, batch_size=5000, shuffle=False)


def KNN(train_data, train_labs, test_data, test_labs):
    # Log time
    start_time = time.time()

    KNN = KNeighborsClassifier(n_neighbors=200)

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.fit_transform(test_data)

    KNN.fit(train_data, train_labs)
    predicted = KNN.predict(test_data)

    # In order to pretty print output
    print('KNN Accuracy', accuracy_score(predicted, test_labs)*100, '%')

    utils.test_best_knn(KNN,train_data,train_labs,test_data,test_labs)

    print("--- %s seconds ---" % (time.time() - start_time))
    
    confmatrix = confusion_matrix(predicted, test_labs)
    plt.subplots(figsize=(6, 6))
    sns.heatmap(confmatrix, annot=True, fmt=".1f", linewidths=1.5)
    plt.savefig('pics/confusion_matrix.png')
    

def test_data():

    train_imgs, train_labs = next(iter(knn_train_loader))
    test_imgs, test_labs = next(iter(knn_test_loader))

    train_codes, _ = model(train_imgs.to(device))
    print('latent layer', train_codes.shape)
    test_codes, _ = model(test_imgs.to(device))

    KNN(train_codes.flatten(1).detach().cpu().numpy(), train_labs,
        test_codes.flatten(1).detach().cpu().numpy(), test_labs)


test_data() 

