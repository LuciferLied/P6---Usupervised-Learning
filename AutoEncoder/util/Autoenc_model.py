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
            nn.Linear(16, 2),
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
    
# Model structure
class CNN_AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 2)
        
        # Decoder
        self.t_fc1 = nn.Linear(2, 120)
        self.t_fc2 = nn.Linear(120, 400)
        
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 10)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 10)

    def forward(self, x):
        
        # Encoder
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        codes = x
        # Decoder
        x = torch.relu(self.t_fc1(x))
        x = torch.relu(self.t_fc2(x))
        unflatten = nn.Unflatten(1, (4, 10, 10))
        x = unflatten(x)
        x = torch.relu(self.t_conv1(x))
        x = torch.sigmoid(self.t_conv2(x))
        decoded = x
        
        return codes, decoded