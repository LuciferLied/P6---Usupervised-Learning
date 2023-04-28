import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.utils.data as data
from util import Models as Model
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torchvision.datasets import CIFAR10

# set device
if torch.cuda.is_available():
    print('Using GPU')
    device = torch.device('cuda')
else:
    print('Using CPU')
    device = torch.device('cpu')

# Settings

batch_size = 128
epochs = 1
lr = 0.001
feature_dim = 128


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
    # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    # transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


train_data = CIFAR10Pair(root='data', train=True, transform=train_transform, download=True)
augmeted_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True,drop_last=True)

model = Model.Res18(feature_dim)
model.to(device)
name = model.__class__.__name__

print('running: ',name, 'on', device, 'with epochs:', epochs, ' and feature dim:', feature_dim)
# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=1e-6)

temperature = 0.5
# Train
def train():
    for epoch in range(epochs):
        total_loss, total_num, train_bar = 0.0, 0, tqdm(augmeted_loader)
        for pics1, pics2, labels in train_bar:
            pics1 = pics1.to(device)
            pics2 = pics2.to(device)

            # Forward
            _, out_1 = model(pics1)
            _, out_2 = model(pics2)
            
            out_1 = torch.flatten(out_1, start_dim=1)
            out_2 = torch.flatten(out_2, start_dim=1)

            out = torch.cat([out_1, out_2], dim=0)
            # [2*B, 2*B]
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
            # [2*B, 2*B-1]
            sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

            # compute loss
            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_num += batch_size
            total_loss += loss.item() * batch_size
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch + 1, epochs, total_loss / total_num))

train()

print('Finished Training')
print('Saving model as: ', 'trained_models/{}_{}_{}.pth'.format(name, epochs, lr))
# Save
torch.save(model, 'trained_models/{}_{}_{}.pth'.format(name, epochs, lr))