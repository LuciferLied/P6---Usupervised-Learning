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
        self.con2 = nn.LazyConv2d(4, 3)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Decoder
        self.t_con1 = nn.LazyConvTranspose2d(6, 4, stride=2)
        self.t_con2 = nn.LazyConvTranspose2d(1, 4)
        self.t_l1 = nn.LazyLinear(256)
        self.t_l2 = nn.LazyLinear(784)
        
    def forward(self, x):
        # Encoder
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = x.view(-1, 1, 12, 12)
        x = torch.relu(self.con1(x))
        x = self.pool(x)
        x = torch.sigmoid(self.con2(x))
        codes = x
        
        # Decoder
        x = torch.relu(self.t_con1(x))
        x = torch.relu(self.t_con2(x))
        x = torch.flatten(x, 2)
        x = torch.relu(self.t_l1(x))
        x = torch.sigmoid(self.t_l2(x))
        
        return codes, x

class Cifar(nn.Module):
    def __init__(self):
        super(Cifar, self).__init__()
        
        # Encoder
        self.l1 = nn.LazyLinear(2000)
        self.l2 = nn.LazyLinear(900)
        self.con1 = nn.LazyConv2d(5, 3)
        self.con2 = nn.LazyConv2d(3, 3)
        
        # Decoder
        self.t_con1 = nn.LazyConvTranspose2d(4, 3)
        self.t_con2 = nn.LazyConvTranspose2d(1, 3)
        self.t_l1 = nn.LazyLinear(2000)
        self.t_l2 = nn.LazyLinear(3072)
        
    def forward(self, x):
        # Encoder
        x = torch.flatten(x, 1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = x.view(-1, 1, 30, 30)
        x = torch.relu(self.con1(x))
        x = self.pool(x)
        x = torch.relu(self.con2(x))

        codes = x
        
        # Decoder
        x = torch.relu(self.t_con1(x))
        x = torch.relu(self.t_con2(x))
        x = torch.flatten(x, 2)
        x = torch.relu(self.t_l1(x))
        x = torch.sigmoid(self.t_l2(x))
        x = x.view(-1, 3, 32, 32)
        
        return codes, x

class LinearCifar(nn.Module):
    def __init__(self):
        super(LinearCifar, self).__init__()
        
        # Encoder
        self.l1 = nn.LazyLinear(2000)
        self.l2 = nn.LazyLinear(1300)
        self.l3 = nn.LazyLinear(800)
        
        # Decoder
        self.t_l1 = nn.LazyLinear(1300)
        self.t_l2 = nn.LazyLinear(2000)
        self.t_l3 = nn.LazyLinear(3072)
        
    def forward(self, x):
        # Encoder
        x = torch.flatten(x, 1)
        x = torch.relu(self.l1(x))
        # x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))

        codes = x
        
        # Decoder
        # x = torch.relu(self.t_l1(x))
        x = torch.relu(self.t_l2(x))
        x = torch.sigmoid(self.t_l3(x))
        x = x.view(-1, 3, 32, 32)
        
        return codes, x

# Make autoencoder for cifar 10
class Cifar_AutoEncoder(nn.Module):
    def __init__(self):
        super(Cifar_AutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 32, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 8, 3),
            nn.MaxPool2d(2, 2)            
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(8, 32, 3),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 3, 3, padding='same'),
            nn.Sigmoid()
        )
        
    def forward(self, inputs):
        codes = self.encoder(inputs)
        decoded = self.decoder(codes)
        
        return codes, decoded

# Make autoencoder for cifar 10
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.LazyConv2d(12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.LazyConv2d(24, 4, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.LazyConv2d(48, 4, stride=2, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.encoder(x)
        return x
        
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.LazyConvTranspose2d(24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.LazyConvTranspose2d(12, 4,stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.LazyConvTranspose2d(3, 4,stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.decoder(x)
        return x
    