import torch
from util import utils as util
import torch.utils.data as data
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from util import Models as Model


# Test
def test(modelPath, test_set, device):
    

    # Load model
    model = torch.load(modelPath)
    model.to(device)
    model.eval()
    #print('model',model)




    test_loader = data.DataLoader(test_set, batch_size=10000, shuffle=False)
        
    with torch.no_grad():
        for pics, labels in test_loader:
 
            pics = pics.to(device)
            pics = torch.flatten(pics, 1)
            
            # Forward
            code, outputs = model(pics)
            
            code = torch.flatten(code, 1)
            #print('code',code.shape)
            code = code.cpu()

            # random_state=0 for same seed in kmeans
            # clusters = KMeans(n_clusters=50, n_init='auto',).fit(code)
            # KNN = KNeighborsClassifier(n_neighbors=1).fit(code, labels)
            kmeans = KMeans(n_clusters=50, n_init='auto').fit(code)

    labels = test_set.targets
    labels = np.array(labels)
    ref_labels = util.retrieveInfo(kmeans.labels_, labels)
    num_predicted = util.assignPredictions(kmeans.labels_, ref_labels)
    accuracy = util.computeAccuracy(num_predicted, labels)

    accuracy = round(accuracy * 100,2)
    #print('Accuracy',accuracy, '%')
    
    return accuracy