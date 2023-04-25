import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from util import Models as Model
from util import utils
from tqdm import tqdm

# set device
if torch.cuda.is_available():
    print('Using GPU')
    device = torch.device('cuda')
else:
    print('Using CPU')
    device = torch.device('cpu')

# Settings
setting = {
    'batch_size': 256,
    'epochs': 5,
    'lr': 0.001
}

from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10


class CIFAR10Pair(CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

# DataLoader
train_set = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
train_loader = data.DataLoader(train_set, batch_size=setting['batch_size'], shuffle=True)

train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.train_transform, download=True)
augmeted_loader = data.DataLoader(train_data, batch_size=setting['batch_size'], shuffle=True, pin_memory=True,drop_last=True)

model = Model.simCLR()
model.to(device)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=setting['lr'],weight_decay=1e-5)
criteria = nn.MSELoss()

temperature = 0.5
# Train
def train():
    for epoch in range(setting['epochs']):
        total_loss, total_num, train_bar = 0.0, 0, tqdm(augmeted_loader)
        for pics1, pics2, labels in train_bar:
            pics1 = pics1.to(device)
            pics2 = pics2.to(device)

            # Forward
            codes, out_1 = model(pics1)
            codes, out_2 = model(pics2)
            
            out_1 = torch.flatten(out_1, start_dim=1)
            out_2 = torch.flatten(out_2, start_dim=1)

            out = torch.cat([out_1, out_2], dim=0)
            # [2*B, 2*B]
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * setting['batch_size'], device=sim_matrix.device)).bool()
            # [2*B, 2*B-1]
            sim_matrix = sim_matrix.masked_select(mask).view(2 * setting['batch_size'], -1)

            # compute loss
            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_num += setting['batch_size']
            total_loss += loss.item() * setting['batch_size']
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, setting['epochs'], total_loss / total_num))

train()

# Save
torch.save(model, 'Cif10.pth')