# Model structure
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
from torchvision.models.mobilenetv2 import mobilenet_v2
from torchvision.models.squeezenet import squeezenet1_1


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
            nn.AvgPool2d(2, 2)
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

class Res18(nn.Module):
    def __init__(self, feature_dim=128):
        super(Res18, self).__init__()

        self.encoder = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.encoder.append(module)

        # encoder
        self.encoder = nn.Sequential(*self.encoder)
        # projection head
        self.g = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True)
        )

    def forward(self, x):
        
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
    
class simCLR(nn.Module):
    def __init__(self, dataset, feature_dim=128):
        super(simCLR, self).__init__()

        self.encoder = []
        if dataset == 'CIFAR10':
            for name, module in resnet18().named_children():
                if name == 'conv1':
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    self.encoder.append(module)
                    
        if dataset == 'MNIST':
            for name, module in resnet18().named_children():
                if name == 'conv1':
                    module = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    self.encoder.append(module)

        # encoder
        self.encoder = nn.Sequential(*self.encoder)
        # projection head
        self.g = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True)
        )

    def forward(self, x):
        
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

# simCLR with MOBILENET V2 as base encoder
class simCLRMobile(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_encoder = mobilenet_v2()
        self.projection_head = nn.Sequential(
            nn.LazyLinear(1280,bias=False),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),
            nn.Linear(1280, 128),
        )
        
    def forward(self, x):
        x = self.base_encoder(x)
        codes = torch.flatten(x, 1)
        out = self.projection_head(codes)
        return F.normalize(codes,dim=1), F.normalize(out, dim=1)    
    
# simCLR with Linear as base encoder
class simCLRLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*32*32, 1000),
            nn.ReLU(),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )
    
    def forward(self, x):
        x = self.base_encoder(x)
        codes = torch.flatten(x, 1)
        out = self.projection_head(codes)
        return F.normalize(codes,dim=1), F.normalize(out, dim=1)

# simCLR with small convolutional network as base encoder
class simCLRSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 8, 3),
            nn.AvgPool2d(2, 2)
        )
        self.projection_head = nn.Sequential(
            nn.LazyLinear(8,bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 128),
        )
        
    def forward(self, x):
        x = self.base_encoder(x)
        codes = torch.flatten(x, 1)
        out = self.projection_head(codes)
        return F.normalize(codes,dim=1), F.normalize(out, dim=1)