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
    'epochs': 16,
    'lr': 0.001
}

Encoder = Model.Encoder()
Encoder.to(device)
Decoder = Model.Decoder()
Decoder.to(device)
# Optimizer and loss function
optimizer = torch.optim.Adam(list(Encoder.parameters()) + list(Decoder.parameters()), lr=setting['lr'])
criteria = nn.MSELoss()

# Train
def train():
    for epoch in range(setting['epochs']):
        for pics, labels in train_loader:
            pics = pics.to(device)

            # Forward
            recon = Decoder(Encoder(pics))

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
torch.save(Encoder, 'Enc.pth')
torch.save(Decoder, 'Dec.pth')