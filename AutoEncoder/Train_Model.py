import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets
from torchvision.transforms import ToTensor
import time
from tqdm import tqdm 
from util import Models as Model
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from util import utils

start = time.time()
# set device
if torch.cuda.is_available():
    print('Using GPU')
    dtype = torch.float32
    device = torch.device('cuda')
else:
    print('Using CPU')
    device = torch.device('cpu')

# Settings
epochs = 30
batch_size = 256
lr = 0.001
neighbors_cluster = 20

# DataLoader
train_set = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True,drop_last=True)

# DataLoader
test_set = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

name = train_set.__class__.__name__
# Optimizer and loss function
model = Model.AE_ResNet(name)
print(model)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = nn.MSELoss()

# Train
for epoch in range(epochs):
    total_loss, total_num, train_bar = 0.0, 0, tqdm(train_loader)
    for pics, labels in train_bar:
        pics = pics.to(device)
        
        # Forward
        codes, decoded = model(pics)

        if name == 'CIFAR10':
            decoded = decoded.view(-1, 3, 32, 32)
        if name == 'MNIST':
            decoded = decoded.view(-1, 1, 28, 28)
        # Backward
        optimizer.zero_grad()
        
        loss = loss_function(decoded, pics)
        
        loss.backward()
        optimizer.step()
        total_num += batch_size
        total_loss += loss.item() * batch_size
        # Show progress
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch + 1, epochs, total_loss / total_num))
    
def KNN(train_data, train_labs, test_data, test_labs):
    
    KNN = KNeighborsClassifier(n_neighbors=20)
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.fit_transform(test_data)

    KNN.fit(train_data, train_labs)
    predicted = KNN.predict(test_data)
    
    accuracy = accuracy_score(predicted, test_labs)
    
    return accuracy*100

def kmeans(train_data, train_labs):
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    kmeans = KMeans(n_clusters=20, n_init='auto').fit(train_data)
    
    labels = train_labs
    # cast to int
    labels = np.array(labels)
    labels = labels.astype(int)
    
    labels = np.array(labels)
    ref_labels = utils.retrieveInfo(kmeans.labels_, labels)
    num_predicted = utils.assignPredictions(kmeans.labels_, ref_labels)
    accuracy = utils.computeAccuracy(num_predicted, labels)

    return accuracy*100

def test(model, train_loader, test_loader, device):
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
        
    knn_acc = KNN(train_codes, train_labs, test_codes, test_labs)
    kmeans_acc = kmeans(train_codes,train_labs)
    
    return knn_acc, kmeans_acc



knn_acc, kmeans_acc = test(model, train_loader, test_loader, device)
print('KNN Accuracy: {:.2f} %'.format(knn_acc))
print('KMeans Accuracy: {:.2f} %'.format(kmeans_acc))

print('Time: ', time.time() - start)
