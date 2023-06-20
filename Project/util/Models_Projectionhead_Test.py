# Model structure
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
from torchvision.models.mobilenetv2 import mobilenet_v2

class simCLR_proj(nn.Module):
    def __init__(self, dataset, feature_dim=128):
        super(simCLR_proj, self).__init__()

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
            nn.Linear(512, feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

class simCLR_proj2(nn.Module):
    def __init__(self, dataset, feature_dim=128):
        super(simCLR_proj2, self).__init__()

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
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

class simCLR_proj3(nn.Module):
    def __init__(self, dataset, feature_dim=128):
        super(simCLR_proj3, self).__init__()

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
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
    
class simCLR_proj4(nn.Module):
    def __init__(self, dataset, feature_dim=128):
        super(simCLR_proj4, self).__init__()

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
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
    
class simCLR_proj5(nn.Module):
    def __init__(self, dataset, feature_dim=128):
        super(simCLR_proj5, self).__init__()

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
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 384),
            nn.ReLU(),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)