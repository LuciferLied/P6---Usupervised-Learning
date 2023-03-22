# https://clay-atlas.com/us/blog/2021/08/03/machine-learning-en-introduction-autoencoder-with-pytorch/
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets
from torchvision.transforms import ToTensor
import time 
from util import Models as Model

start = time.time()
# set device
if torch.cuda.is_available():
    print('Using GPU')
    dtype = torch.float32
    device = torch.device('cuda:0')
else:
    print('Using CPU')
    device = torch.device('cpu')


# Settings
epochs = 5
batch_size = 128
lr = 0.01

# DataLoader
train_set = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Optimizer and loss function
model = Model.AutoEncoder()
print(model)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = nn.MSELoss()

# Train
for epoch in range(epochs):
    for data, labels in train_loader:
        inputs = data.view(-1, 784)
        inputs = inputs.to(device)
        
        # Forward
        codes, decoded = model(inputs)

        # Backward
        optimizer.zero_grad()
                
        loss = loss_function(decoded, inputs)
        loss.backward()
        optimizer.step()

    # Show progress
    print('[{}/{}] Loss:'.format(epoch+1, epochs), loss.item())

print('Finished Training using', device)
print('Time: ', time.time() - start)

# Save
torch.save(model, 'autoencoder.pth')