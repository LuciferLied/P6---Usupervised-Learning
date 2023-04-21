from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision
import time
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from torchvision.transforms import ToTensor

# set device
if torch.cuda.is_available():
    print('Using GPU')
    device = torch.device('cuda')
else:
    print('Using CPU')
    device = torch.device('cpu')

setting = {
    'epochs': 25,
    'lr': 0.001,
    'batch_size': 256,
}

train_labeled_dataset = STL10(
    root="data",
    split="train",
    download=True,
    transform=ToTensor()
)
train_unlabeled_dataset = STL10(
    root="data",
    split="unlabeled",
    download=True,
    transform=ToTensor()
)
test_labeled_dataset = STL10(
    root="data",
    split="test",
    download=True,
    transform=ToTensor()
)

# Train labeled is 5000 images
# Train shape is (5000, 3, 96, 96)
train_labeled_dataloader = DataLoader(dataset=train_labeled_dataset,batch_size=setting["batch_size"],shuffle=True) #500 of each class

# Train unlabeled is 100,000 images
# Train shape is (100000, 3, 96, 96)
train_unlabeled_dataloader = DataLoader(dataset=train_unlabeled_dataset,batch_size=15000,shuffle=True)

# Test labeled is 8000 images
# Test shape is (8000, 3, 96, 96)
test_labeled_dataloader = DataLoader(dataset=test_labeled_dataset,batch_size=setting["batch_size"])

classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

class SupervisedModel():
    def __init__(self, encoder):
        self.model = encoder.to(device)

    @staticmethod
    def count_correct(y_pred, y_true):
        # Returns the number of correct predictions
        preds = torch.argmax(y_pred, dim=1)
        return (preds == y_true).float().sum()

    def validate_model(self,loss_fn, dataloader):
        # Validates given model with given loss function on given DataLoader
        loss = 0
        correct = 0
        all = 0
        with torch.no_grad():
            for pics, labels in dataloader:
                pics = pics.to(device)
                labels = labels.to(device)
                #all is the number of images
                labels = labels.long()
                y_pred = self.model(pics)
                all += len(y_pred)
                #loss is the sum of the loss of each image
                loss += loss_fn(y_pred, labels)
                
                #Correct is the number of correct predictions
                correct += SupervisedModel.count_correct(y_pred, labels)
                break
                
        #Return the average loss and the accuracy
        return loss / all, correct / all

    def train_model(self, optimiser, loss_fn, train_dl):
        # Trains given model with given loss function on given DataLoader
        self.model.train()

        for pics, labels in train_dl:
            pics = pics.to(device)
            labels = labels.to(device)
            labels = labels.long()
            y_pred = self.model(pics)
            # loss is the sum of the loss of each image
            loss = loss_fn(y_pred, labels)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

        self.model.eval()

    def fit(self, loss_fn, train_dl, test_dl, epochs):
        
        # Trains model and validates on training and validation data
        results = {"train_loss": [], "train_acc": [], "val_loss": [],
                   "val_acc": []}
        optimiser = optim.AdamW(self.model.parameters())

        for epoch in range(epochs):
            start = time.time()
            self.train_model(optimiser, loss_fn, train_dl)
            train_loss, train_acc = self.validate_model(loss_fn, train_dl)
            val_loss, val_acc = self.validate_model(loss_fn, test_dl)
            print(f"Epoch {epoch + 1}: train loss = {train_loss:.3f} "
             f"(acc: {train_acc:.3f}), validation loss = {val_loss:.3f} "
             f"(acc: {val_acc:.3f}), time {time.time() - start:.1f}s")
            results["train_loss"].append(float(train_loss))
            results["train_acc"].append(float(train_acc))
            results["val_loss"].append(float(val_loss))
            results["val_acc"].append(float(val_acc))

        return results
    
    # self label
    def self_label(self, train_dl):
        predicts = []
        predicts = torch.tensor(predicts)
        pictures = []
        pictures = torch.tensor(pictures)
        
        data = torch.utils.data.TensorDataset()
        
        print("self label start")
        start = time.time()
        batch = 0
        for pics, labels in train_dl:
            pics = pics.to('cpu')
            labels = labels.to('cpu')
            print(f"batch {batch}")
            batch += 1
            
            y_pred = self.model(pics)
            preds = torch.argmax(y_pred, dim=1)
            preds = preds.int()
            print(preds)
            preds = preds.to('cpu')
            pics = pics.to('cpu')
            predicts = torch.cat((predicts, preds), 0)
            pictures = torch.cat((pictures, pics), 0)

        print(f"time {time.time() - start:.1f}s")

        # Make a new dataset with the pics and the predictions with batch size 128
        data = torch.utils.data.TensorDataset(pictures, predicts)
        data = DataLoader(dataset=data,batch_size=setting["batch_size"])
        return data

def train_model():
    print('trainig model')
    encoder = torchvision.models.resnet18()
    encoder.fc = nn.Linear(512, 10) # Resnet18 is created for 1000 classses so we need to change the last layer
    baseline_model = SupervisedModel(encoder)

    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    torch.save(baseline_model, 'semi.pth')
    
    return baseline_model.fit(loss_fn, train_labeled_dataloader, test_labeled_dataloader, setting["epochs"])

def self_label(data):
    # Load the model
    with torch.no_grad():
        self_label_model = torch.load('semi.pth') 
        self_label_model.model.eval()
        self_label_model.model.to('cpu')
        
        self_labeled = self_label_model.self_label(data)

        torch.save(self_labeled, 'self_labeled.pt')
                
        for pics, labels in self_labeled:
            plt.imshow(pics[7].cpu().squeeze().numpy().transpose(1, 2, 0))
            # Set the title to the class name
            title = labels[7].to(int)
            plt.title(classes[title])
            plt.savefig('pics/original.png')
            break
        return 
    
def train_again():
    # Load the model
    print('Self labeled model')
    encoder = torchvision.models.resnet18()
    encoder.fc = nn.Linear(512, 10)
    self_label_model2 = SupervisedModel(encoder)
    
    # self_label_model = torch.load('semi.pth')
    # self_label_model.model.to(device)
    self_labeled = torch.load('self_labeled.pt')

    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    return self_label_model2.fit(loss_fn, self_labeled, test_labeled_dataloader, setting["epochs"])


history_baseline = train_model()
self_label(train_unlabeled_dataloader)
train_again()


