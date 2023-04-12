import numpy as np
import torch
import torch.nn as nn
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

setting = {
    'epochs': 7,
    'lr': 0.001
}

model = Model.Auto_CNN()
model.to(device)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=setting['lr'])
criteria = nn.MSELoss()
criteria1 = nn.CrossEntropyLoss()


def KNN(TextString,x_train,y_train,x_test,y_test):

    print('x_train shape: ',x_train.shape)
    print('x_test shape: ',x_test.shape)
 
    clf = KNeighborsClassifier(n_neighbors=3)
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
        arr = []
        arr = torch.tensor(arr)
        yarr = []
        yarr = torch.tensor(yarr)

        for pics, y_train in train_loader:
            pics = pics.to(device)
            x_train, decoded = model(pics)
            x_train = x_train.to('cpu')
            x_train = torch.flatten(x_train, 1)
            
            yarr = torch.cat((yarr, y_train), 0)
            arr = torch.cat((arr, x_train), 0)
        
    with torch.no_grad():
        for pics, labels in test_loader:
            pics = pics.to(device)
            codes, decoded = model(pics)
            codes = codes.to('cpu')
            codes = torch.flatten(codes, 1)
            
        x_train = np.array(arr)
        x_train = x_train[:-1]
        y_train = np.array(yarr)
        y_train = y_train[:-1]
        x_test = np.array(codes)
        y_test = np.array(labels)
        
    KNN("CIFAR10",x_train, y_train, x_test, y_test)

# Train
def train():
    for epoch in range(setting['epochs']):
        for pics, labels in train_loader:
            pics = pics.to(device)
            
            # Forward
            codes, decoded = model(pics)
            
            # Backward
            optimizer.zero_grad()
            loss = criteria(decoded, pics)            
            loss.backward()
            optimizer.step()
            
            save_pic = pics
            save_dec = decoded
            
        # Show progress
        print('[{}/{}] Loss:'.format(epoch+1, setting['epochs']), loss.item())

        plt.imshow(save_pic[1].cpu().squeeze().numpy().transpose(1, 2, 0))
        plt.savefig('pics/original.png')
        
        save_dec = save_dec[1].view(-1, 3, 32, 32)
        plt.imshow(save_dec.cpu().squeeze().detach().numpy().transpose(1, 2, 0))
        plt.savefig('pics/reconstructed.png')
        
    test_data()

train()

def test():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            images = nn.functional.normalize(images)
            
            encoded, classed, decoded = model(images)
            
            _, predicted = torch.max(classed.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
# Test
def test1():
    with torch.no_grad():
        for pics, labels in test_loader:
            
            pics = pics.to(device)
            pics = nn.functional.normalize(pics)
            # Forward
            encoded, classed, decoded = model(pics)
            code = torch.flatten(decoded, 1)
            code = code.to('cpu')

            # random_state=0 for same seed in kmeans
            clusters = KMeans(n_clusters=16, n_init='auto',).fit(code)
        
            # labels = test_set.targets
            # labels = np.array(labels)
            ref_labels = util.retrieveInfo(clusters.labels_, labels)
            num_predicted = util.assignPredictions(clusters.labels_, ref_labels)
            accuracy = util.computeAccuracy(num_predicted, labels)

            # print('Ref_labels',ref_labels)
            # print('labels',labels[0:20])
            # print('num_predicted',num_predicted[0:20])
            # print(model)
            # Round accuracy to 2 decimals
            accuracy = round(accuracy * 100,2)
            print('Accuracy',accuracy, '%')

# test1()