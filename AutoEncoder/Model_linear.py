# https://clay-atlas.com/us/blog/2021/08/03/machine-learning-en-introduction-autoencoder-with-pytorch/
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as T
import time 
import matplotlib.pyplot as plt
import numpy as np
import warnings
from tqdm import tqdm
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

class Resnet18(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Resnet18, self).__init__()

        # encoder
        self.f = Res18().encoder
        # classifier
        self.fc = nn.Linear(512, num_class, bias=True)
        self.model = torch.load(pretrained_path, map_location='cpu')

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out

# Optimizer and loss function
model = Resnet18(num_class = 10, pretrained_path = 'penis')
# print(model)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()
train = False
# Train
for epoch in range(5):
  if epoch == 5:
    train = True
  total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(train_loader)
  with torch.no_grad():
      for data, target in data_bar:
          data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
          out = model(data)
          loss = loss_function(out, target)
          if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
          total_num += data.size(0)
          total_loss += loss.item() * data.size(0)
          prediction = torch.argsort(out, dim=-1, descending=True)
          total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

          data_bar.set_description('ACC :{}'.format(total_correct_1 / total_num * 100))

print('Finished Training using', device)
print('Time: ', time.time() - start)

