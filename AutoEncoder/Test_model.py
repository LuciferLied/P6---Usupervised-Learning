import torch
from util import utils as util
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


# Settings
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# set device
if torch.cuda.is_available():
    print('Using GPU')
    dtype = torch.float32
    device = torch.device('cuda:0')
else:
    print('Using CPU')
    device = torch.device('cpu')

# Show images
def show_images(images):
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

    for index, image in enumerate(images):
        plt.subplot(sqrtn, sqrtn, index+1)
        plt.imshow(image.reshape(28, 28))
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
model.to(device)
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
        
        inputs = inputs.to(device)
        code, outputs = model(inputs)

        # Make code and outputs numpy array
        code = code.to('cpu')
        # random_state=0 for same seed in kmeans
        #clusters = MiniBatchKMeans(n_clusters=20, n_init='auto',).fit(data)
        clusters = KMeans(n_clusters=10, n_init='auto',).fit(code)



labels = test_set.targets
unique_labels = len(np.unique(labels))
ref_labels = util.retrieveInfo(clusters.labels_, labels)
num_predicted = util.assignPredictions(clusters.labels_, ref_labels)


accuracy = util.computeAccuracy(num_predicted, labels)

print('Ref_labels',ref_labels)
print('labels',labels[0:20])
print('num_predicted',num_predicted[0:20])
print('Accuracy',accuracy)

def plot(ref_labels):
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x, y, c=clusters.labels_, cmap='rainbow')
    cluster, numbers = zip(*ref_labels.items())
    plt.colorbar(ticks=numbers)
    plt.savefig('Pics/clusters.png')
    
    
    # Plotting  the clusters
    # plt.scatter(cluster, numbers)
    # plt.xlabel("Cluster")
    # plt.ylabel("Number")
    # plt.savefig('Pics/numbers.png')

# plot(ref_labels)