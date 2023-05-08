# https://clay-atlas.com/us/blog/2021/08/03/machine-learning-en-introduction-autoencoder-with-pytorch/
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as T
import time 
from util import Models as Model
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

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
epochs = 5
batch_size = 256
lr = 0.001

# DataLoader
train_set = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)


# Optimizer and loss function
model = Model.Cifar_AutoEncoder()
print(model)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = nn.MSELoss()

# Train
for epoch in range(epochs):
    for pics, labels in train_loader:
        pics = pics.to(device)
        pics1 = pics
        
        #greyscale pics
        # pics = T.Grayscale()(pics)
        # pics = torch.flatten(pics, 1)

        # Forward
        codes, decoded = model(pics)

        decoded1 = decoded
        # decoded = torch.flatten(decoded, 1)
        # Backward
        optimizer.zero_grad()
        
        loss = loss_function(decoded, pics)
        
        loss.backward()
        optimizer.step()
    print('codes', codes.shape)
    # Show progress
    print('[{}/{}] Loss:'.format(epoch+1, epochs), loss.item())

    plt.imshow(pics1[1].cpu().squeeze().numpy().transpose(1, 2, 0))
    # plt.imshow(pics1[1].cpu().squeeze().numpy())
    plt.savefig('pics/original.png')
    decoded1 = decoded1[1].view(-1, 3, 32, 32)
    plt.imshow(decoded1.cpu().squeeze().detach().numpy().transpose(1, 2, 0))
    # plt.imshow(decoded1.cpu().squeeze().detach().numpy())
    plt.savefig('pics/reconstructed.png')



print('Finished Training using', device)
print('Time: ', time.time() - start)

# Save
torch.save(model, 'autoencoder.pth')