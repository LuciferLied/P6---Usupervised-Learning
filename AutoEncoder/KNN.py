import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from util import Models as Model
import warnings
warnings.filterwarnings("ignore")


# set device
if torch.cuda.is_available():
    print('Using GPU')
    # dtype = torch.float32
    device = torch.device('cuda')
else:
    print('Using CPU')
    device = torch.device('cpu')

# Settings
setting = {
    'batch_size': 256,
    'epochs': 14,
    'lr': 0.001
}


# DataLoader
train_set = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

train_loader = data.DataLoader(train_set, batch_size=setting['batch_size'], shuffle=True)

model = Model.Cifar_AutoEncoder()
model.to(device)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=setting['lr'])
criteria = nn.MSELoss()

# Train
def train():
    for epoch in range(setting['epochs']):
        for pics, labels in train_loader:
            pics = pics.to(device)

            # Forward
            codes, recon = model(pics)

            # Backward
            optimizer.zero_grad()
            loss = criteria(recon, pics)            
            loss.backward()
            optimizer.step()
            
            save_pic = pics
            save_dec = recon
        
        # Show progress
        print('[{}/{}] Loss:'.format(epoch+1, setting['epochs']), loss.item())

        plt.imshow(save_pic[1].cpu().squeeze().numpy().transpose(1, 2, 0))
        plt.savefig('pics/original.png')
        
        save_dec = save_dec[1].view(-1, 3, 32, 32)
        plt.imshow(save_dec.cpu().squeeze().detach().numpy().transpose(1, 2, 0))
        plt.savefig('pics/reconstructed.png')

train()

# Save
torch.save(model, 'Cif10.pth')