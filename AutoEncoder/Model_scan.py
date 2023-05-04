import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import CIFAR10
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# set device
if torch.cuda.is_available():
    print('Using GPU')
    dtype = torch.float32
    device = torch.device('cuda')
else:
    print('Using CPU')
    device = torch.device('cpu')

# Settings
batch_size = 256
epochs = 10
lr = 0.001
feature_dim = 128

dataset = CIFAR10
data_name = dataset.__name__

train_transform = transforms.Compose([
    # transforms.RandomResizedCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# DataLoader
train_dataset = dataset(
    root="data",
    train=True,
    download=False,
    transform=train_transform,
)

train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)



class SCAN(nn.Module):
    """
    A PyTorch implementation of the SCAN algorithm for unsupervised classification.

    Args:
        backbone (torch.nn.Module): The feature extraction backbone.
        feature_dim (int): The dimensionality of the feature space.
        num_clusters (int): The number of clusters to use.
        use_softmax (bool): Whether to apply softmax to the cluster assignments.
    """
    def __init__(self, feature_dim=128, num_clusters=10, use_softmax=True):
        super().__init__()
        self.backbone = torch.load('trained_models/Res18_CIFAR10_30_0.001.pth', map_location=torch.device(device))
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters
        self.use_softmax = use_softmax

        self.fc = nn.Linear(feature_dim, num_clusters)
        nn.init.normal_(self.fc.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        # Extract the features from the backbone
        _, features = self.backbone(x)

        # Compute the cluster assignments
        if self.use_softmax:
            assignments = nn.functional.softmax(self.fc(features), dim=1)
        else:
            assignments = self.fc(features)

        return features, assignments

def train_scan(model, dataloader, num_clusters, device):
    """
    Train the SCAN model.

    Args:
        model (SCAN): The SCAN model to train.
        dataloader (torch.utils.data.DataLoader): The dataloader to use for training.
        num_clusters (int): The number of clusters to use.
        device (torch.device): The device to use for training.

    Returns:
        tuple: A tuple containing the trained model and the final cluster assignments.
    """
    # Set up the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    clustering_criterion = nn.KLDivLoss(reduction="batchmean")

    # Train the model
    model.train()
    for epoch in range(epochs):
        total_loss, total_num, train_bar = 0.0, 0, tqdm(dataloader)
        for x, y in train_bar:
            x = x.to(device)

            # Compute the features and cluster assignments
            features, assignments = model(x)

            # Compute the cluster centers using KMeans
            kmeans = KMeans(n_clusters=num_clusters, n_init=20)
            cluster_centers = kmeans.fit_predict(assignments.detach().cpu().numpy())

            # Compute the clustering loss
            one_hot_assignments = torch.eye(num_clusters)[cluster_centers].to(device)
            clustering_loss = clustering_criterion(nn.functional.log_softmax(assignments, dim=1), one_hot_assignments)

            # Backpropagate and update the model
            optimizer.zero_grad()
            clustering_loss.backward()
            optimizer.step()
            
            total_num += batch_size
            total_loss += clustering_loss.item() * batch_size
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch + 1, epochs, total_loss / total_num))
    torch.save(model, 'trained_models/SCAN_CIFAR10_30_0.001.pth')
    # Assign the final clusters based on the learned representations
    with torch.no_grad():
        model.eval()
        all_assignments = []
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            features, assignments = model(x)
            all_assignments.append(assignments)
        final_assignments = torch.cat(all_assignments, dim=0)
        
    print('fin',final_assignments[1].cpu().numpy())
    return model, final_assignments.cpu().numpy()


model = SCAN(feature_dim=feature_dim, num_clusters=10, use_softmax=True).to(device)
model, final_assignments = train_scan(model, train_dataloader, 10, device)

# Calculate the clustering accuracy
y_pred = final_assignments.argmax(1)
print(y_pred.shape)
y_true = np.array(train_dataset.targets)
y_true = y_true[:len(y_pred)]
print(y_true.shape)
acc = accuracy_score(y_true, y_pred)
print("Final clustering accuracy: {:.2f}%".format(acc * 100))