import torch
from torchsummary import summary
from torch import nn
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.utils.data as data
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from util import Models as Model
from util import utils as util

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)
    
class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels//4,
                               kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(
            out_channels//4, out_channels//4, kernel_size=3, stride=2 if downsample else 1, padding=1)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1)

        if self.downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=2 if self.downsample else 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.bn2 = nn.BatchNorm2d(out_channels//4)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = nn.ReLU()(self.bn3(self.conv3(input)))
        input = input + shortcut
        return nn.ReLU()(input)
    
class ResNet(nn.Module):
    def __init__(self, in_channels, resblock, repeat, useBottleneck=False, outputs=1000):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        if useBottleneck:
            filters = [64, 256, 512, 1024, 2048]
        else:
            filters = [64, 64, 128, 256, 512]

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv2_1', resblock(filters[0], filters[1], downsample=False))
        for i in range(1, repeat[0]):
                self.layer1.add_module('conv2_%d'%(i+1,), resblock(filters[1], filters[1], downsample=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv3_1', resblock(filters[1], filters[2], downsample=True))
        for i in range(1, repeat[1]):
                self.layer2.add_module('conv3_%d' % (
                    i+1,), resblock(filters[2], filters[2], downsample=False))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv4_1', resblock(filters[2], filters[3], downsample=True))
        for i in range(1, repeat[2]):
            self.layer3.add_module('conv2_%d' % (
                i+1,), resblock(filters[3], filters[3], downsample=False))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv5_1', resblock(filters[3], filters[4], downsample=True))
        for i in range(1, repeat[3]):
            self.layer4.add_module('conv3_%d'%(i+1,),resblock(filters[4], filters[4], downsample=False))

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(filters[4], outputs)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        # torch.flatten()
        # https://stackoverflow.com/questions/60115633/pytorch-flatten-doesnt-maintain-batch-size
        input = torch.flatten(input, start_dim=1)
        input = self.fc(input)

        return input

resnet18 = ResNet(3, ResBlock, [2, 2, 2, 2], useBottleneck=False, outputs=1000)
resnet18.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# summary(resnet18, (3, 224, 224))

batch_size = 256
epochs = 1
lr = 0.001


# DataLoader
train_set = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(resnet18.parameters(), lr=lr)
loss_function = nn.MSELoss()

model = Model.Resnet_smol().to(device)

def train():
    # Train
    for epoch in range(epochs):
        for pics, labels in train_loader:
            pics = pics.to(device)
            # pics = torch.flatten(pics, 1)

            # Forward
            decoded = resnet18(pics)
            
            decoded = torch.flatten(decoded, 1)
            
            decoded = model(decoded)
            pics = torch.flatten(pics, 1)
            # Backward
            optimizer.zero_grad()
            loss = loss_function(decoded, pics)
            loss.backward()
            optimizer.step()

        # Show progress
        print('[{}/{}] Loss:'.format(epoch+1, epochs), loss.item())


# train()

# DataLoader
test_set = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

test_loader = data.DataLoader(test_set, batch_size=10000, shuffle=False)


# Test
def test():
    with torch.no_grad():
        for pics, labels in test_loader:
            
            pics = pics.to(device)
            # pics = torch.flatten(pics, 1)
            
            # Forward
            code = resnet18(pics)
            code = model(code)
            pics = torch.flatten(pics, 1)
            
            code = code.to('cpu')

            # random_state=0 for same seed in kmeans
            # clusters = MiniBatchKMeans(n_clusters=20, n_init='auto',).fit(code)
            clusters = KMeans(n_clusters=25, n_init='auto',).fit(code)
        print("Done")
        print(code.shape)
        
    # labels = test_set.targets
    # labels = np.array(labels)
    ref_labels = util.retrieveInfo(clusters.labels_, labels)
    num_predicted = util.assignPredictions(clusters.labels_, ref_labels)
    
    accuracy = util.computeAccuracy(num_predicted, labels)

    # print('Ref_labels',ref_labels)
    # print('labels',labels[0:20])
    # print('num_predicted',num_predicted[0:20])
    # print(model)
    
    # Round accuracy to 2 decimals
    accuracy = round(accuracy * 100,2)
    
    
    print('Accuracy',accuracy, '%')

test()