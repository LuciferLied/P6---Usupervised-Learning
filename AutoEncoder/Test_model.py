import torch
from util import utils as util
import torch.utils.data as data
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans


# set device
if torch.cuda.is_available():
    print('Using GPU')
    dtype = torch.float32
    device = torch.device('cuda:0')
else:
    print('Using CPU')
    device = torch.device('cpu')

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
def test():
    with torch.no_grad():
        for data, labels in test_loader:
            
            inputs = data.view(-1, 1, 784)
            inputs = inputs.to(device)
            
            # Forward
            code, outputs = model(inputs)
            
            code = code.to('cpu')
            print(code.shape)
            
            code = torch.flatten(code, 1)

            # random_state=0 for same seed in kmeans
            #clusters = MiniBatchKMeans(n_clusters=20, n_init='auto',).fit(data)
            clusters = KMeans(n_clusters=25, n_init='auto',).fit(code)
        print(code.shape)

    labels = test_set.targets
    ref_labels = util.retrieveInfo(clusters.labels_, labels)
    num_predicted = util.assignPredictions(clusters.labels_, ref_labels)

    accuracy = util.computeAccuracy(num_predicted, labels)

    # print('Ref_labels',ref_labels)
    # print('labels',labels[0:20])
    # print('num_predicted',num_predicted[0:20])
    print(model)
    
    # Round accuracy to 2 decimals
    accuracy = round(accuracy * 100,2)
    
    
    print('Accuracy',accuracy, '%')

test()
