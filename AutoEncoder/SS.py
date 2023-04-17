import torch
import torchvision
import time
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from torchvision.transforms import ToTensor


BATCH_SIZE = 128

train_labeled_dataset = STL10(root="data",split="train", download=True, transform=ToTensor())
train_unlabeled_dataset = STL10(root="data",split="unlabeled", download=True, transform=ToTensor())
test_labeled_dataset = STL10(root="data",split="test", download=True, transform=ToTensor())

train_labeled_dataloader = DataLoader(
    dataset=train_labeled_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
)
train_unlabeled_dataloader = DataLoader(
    dataset=train_unlabeled_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
)
test_labeled_dataloader = DataLoader(
    dataset=test_labeled_dataset,
    batch_size=BATCH_SIZE,
    drop_last=False,
)

device = 'cpu'


class SupervisedModel():
    def __init__(self, encoder):
        self.model = encoder
        #self.model = encoder.cuda()

    @staticmethod
    def count_correct(
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns the number of correct predictions
        """
        preds = torch.argmax(y_pred, dim=1)
        return (preds == y_true).float().sum()


    def validate_model(
        self,
        loss_fn, 
        dataloader: DataLoader
    ):
        """
        Validates given model with given loss function on given DataLoader
        """
        loss = 0
        correct = 0
        all = 0
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch, y_batch
                #X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
                y_pred = self.model(X_batch)
                all += len(y_pred)
                loss += loss_fn(y_pred, y_batch)
                correct += SupervisedModel.count_correct(y_pred, y_batch)
        return loss / all, correct / all


    def train_model(
        self,
        optimiser: optim.Optimizer,
        loss_fn,
        train_dl: DataLoader,
    ):
        """
        Trains given model with given loss function on given DataLoader
        """
        self.model.train()

        for X_batch, y_batch in train_dl:
            X_batch, y_batch = X_batch, y_batch
            #X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
            y_pred = self.model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            loss.backward()
            optimiser.step()
            optimiser.zero_grad() 

        self.model.eval() 


    def fit(
        self,
        loss_fn,
        train_dl: DataLoader,
        val_dl: DataLoader,
        epochs=5
    ) -> dict:
        """
        Trains model and validates on training and validation data
        """
        results = {"train_loss": [], "train_acc": [], "val_loss": [],
                   "val_acc": []}
        optimiser = optim.AdamW(self.model.parameters())  

        for epoch in range(epochs):
            start = time.time()

            self.train_model(optimiser, loss_fn, train_dl)
            train_loss, train_acc = self.validate_model(loss_fn, train_dl) 
            val_loss, val_acc = self.validate_model(loss_fn, val_dl)

            print(f"Epoch {epoch + 1}: train loss = {train_loss:.3f} "
             f"(acc: {train_acc:.3f}), validation loss = {val_loss:.3f} "
             f"(acc: {val_acc:.3f}), time {time.time() - start:.1f}s")
            results["train_loss"].append(float(train_loss))
            results["train_acc"].append(float(train_acc))
            results["val_loss"].append(float(val_loss))
            results["val_acc"].append(float(val_acc))

        return results

def evaluate_baseline():
    encoder = torchvision.models.resnet18()
    encoder.fc = nn.Linear(512, 10) # Resnet18 is created for 1000 classses so we need to change the last layer
    baseline_model = SupervisedModel(encoder)

    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    return baseline_model.fit(loss_fn, train_labeled_dataloader, test_labeled_dataloader)

history_baseline = evaluate_baseline()