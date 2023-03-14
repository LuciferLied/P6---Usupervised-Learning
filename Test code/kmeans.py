#Three lines to make our compiler able to draw:
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy
from sklearn.cluster import KMeans

# # Download training data from open datasets.
# training_data = datasets.MNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor(),
# )

# # Download test data from open datasets.
# test_data = datasets.MNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor(),
# )

# image = test_data[0][0]
# label = test_data[0][1]

# # print('test',image)
# print('test',label)

# image = torch.flatten(image)

x = numpy.random.randint(100,size=(100))
y = numpy.random.randint(100,size=(100))

print('x: ', x.shape)
print('y: ', type(y))

data = list(zip(x, y))

clusters = KMeans(n_clusters=5,n_init='auto')
clusters.fit(data)

plt.scatter(x, y, c=clusters.labels_, cmap='rainbow')
plt.savefig('test.png')