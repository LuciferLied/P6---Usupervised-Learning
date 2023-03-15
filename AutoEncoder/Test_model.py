# coding: utf-8
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from ..KmeansClustering import kmeanFunction as Func


# Settings
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# Show images
def show_images(images):
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

    for index, image in enumerate(images):
        plt.subplot(sqrtn, sqrtn, index+1)
        plt.imshow(image.reshape(28, 28))
        plt.axis('off')

# Show images
def show_images2(images):
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

    for index, image in enumerate(images):
        plt.subplot(sqrtn, sqrtn, index+1)
        plt.imshow(image.reshape(64, 64))
        plt.axis('off')


# Model structure
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, 2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        codes = self.encoder(inputs)
        decoded = self.decoder(codes)
        
        return codes, decoded

# Load model
model = torch.load('autoencoder.pth')
model.eval()

# DataLoader
test_set = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

test_loader = data.DataLoader(test_set, batch_size=10000, shuffle=False)


# Test
with torch.no_grad():
    for x, data in enumerate(test_loader):
        inputs = data[0].view(-1, 28*28)

        plt.savefig('Pics/Original.png')

        code, outputs = model(inputs)
        
        x,y = zip(*code)
        x = np.array(x)
        y = np.array(y)
        
        data = list(zip(x, y))
        
        clusters = KMeans(n_clusters=10, n_init='auto').fit(data)
        
        plt.scatter(x, y, c=clusters.labels_, cmap='rainbow')
        plt.savefig('Pics/clusters.png')



#Associates each cluster with most probable label


labels = test_set.targets

ref_labels = Func.retrieveInfo(clusters.labels_, labels)
predicted_num = Func.assignPredictions(clusters.labels_, ref_labels)
accuracy = Func.computeAccuracy(predicted_num, labels)

print('Ref_labels',ref_labels)
print('labels',labels[0:20])
print('predicted_num',predicted_num[0:20])
print('Accuracy',accuracy)