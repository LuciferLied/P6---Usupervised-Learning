# Model structure
import torch
import torch.nn as nn

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
            nn.Linear(16, 2)
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

class Smol_AutoEncoder(nn.Module):
    def __init__(self):
        super(Smol_AutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.Tanh(),
            nn.Linear(128, 64)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        
        codes = self.encoder(inputs)
        decoded = self.decoder(codes)
        
        return codes, decoded

class Auto_CNN(nn.Module):
    def __init__(self):
        super(Auto_CNN, self).__init__()
        
        # Input: (1 channel, 28x28 pixels) = 784
        # Encoder        
        self.l1 = nn.LazyLinear(256)
        self.l2 = nn.LazyLinear(144)
        self.con1 = nn.LazyConv2d(6, 3)
        self.con2 = nn.LazyConv2d(12, 3)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Decoder
        self.t_con1 = nn.LazyConvTranspose2d(12, 4, stride=2)
        self.t_con2 = nn.LazyConvTranspose2d(1, 5)
        self.t_l1 = nn.LazyLinear(256)
        self.t_l2 = nn.LazyLinear(784)
        
    def forward(self, x):
        # Encoder
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = x.view(-1, 1, 12, 12)
        x = torch.relu(self.con1(x))
        x = torch.sigmoid(self.con2(x))
        codes = x
        
        # Decoder
        x = torch.relu(self.t_con1(x))
        x = torch.relu(self.t_con2(x))
        x = torch.flatten(x, 2)
        x = torch.relu(self.t_l1(x))
        x = torch.sigmoid(self.t_l2(x))
        
        return codes, x
        
    
# Write a model with autoencoder structure with linear and convolutional layers for the mnist dataset 
class Autoencoder_CNN(nn.Module):
    def __init__(self):
        super(Autoencoder_CNN, self).__init__()
        # Encoder
        self.l1 = nn.Conv2d(1, 16, 3, stride=3, padding=1),
        self.pool1 = nn.MaxPool2d(2, stride=2),
        self.l2 = nn.Conv2d(16, 8, 3, stride=2, padding=1),
        self.pool2 = nn.MaxPool2d(2, stride=1)
        
        # Decoder
        self.T_con1 = nn.ConvTranspose2d(8, 16, 3, stride=2),
        self.T_con2 = nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
        self.T_con3 = nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
        
        
    # Forward pass
    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = self.pool1(x)
        x = torch.relu(self.l2(x))
        x = self.pool2(x)

        codes = x
        
        x = torch.relu(self.T_con1(x))
        x = torch.relu(self.T_con2(x))
        x = torch.sigmoid(self.T_con3(x))
        
        return codes, x
