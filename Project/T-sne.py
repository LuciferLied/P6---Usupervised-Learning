import torch
import torch.utils.data as data
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from util import utils
from sklearn.manifold import TSNE

# set device
if torch.cuda.is_available():
    print('Using GPU')
    dtype = torch.float32
    device = torch.device('cuda')
else:
    print('Using CPU')
    device = torch.device('cpu')

#load_model = 'trained_models/simCLR_CIFAR10_500_512_0.001.pth'
load_model = 'trained_models/simCLR_MNIST_500_512_0.001.pth'
# Load model
model = torch.load(load_model, map_location=torch.device(device))
model.to(device)
model.eval()

temp = load_model.split('/')
vars = temp[1].split('_')
name = vars[0]
data_name = vars[1]
epochs = vars[2]
epochs = int(epochs)
lr = vars[3].replace('.pth', '')
lr = float(lr)
batch_size = 256

print('Model:', name,'Trained with dataset:', data_name, 'epochs:', epochs, 'lr:', lr)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

if data_name == 'CIFAR10':
    # DataLoader
    train_set = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=test_transform
    )

    test_set = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=test_transform
    )

if data_name == 'MNIST':
    # DataLoader
    train_set = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=test_transform
    )

    test_set = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=test_transform
    )


train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

mnist_classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Turn numbers into labels
def num_to_label(num):
    return classes[num]


import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import matplotlib as mpl



def KNN(train_data, train_labs, test_data, test_labs):
    
    KNN = KNeighborsClassifier(n_neighbors=20)

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.fit_transform(test_data)

    KNN.fit(train_data, train_labs)
    predicted = KNN.predict(test_data)
                    

    # In order to pretty print output
    print('KNN Accuracy', accuracy_score(predicted, test_labs)*100, '%')
    # utils.test_best_knn(KNN,train_data,train_labs,test_data,test_labs)
    
    # Convert train and test data to numpy arrays
    train_data = np.array(train_data)
    train_labs = np.array(train_labs)
    test_data = np.array(test_data)
    test_labs = np.array(test_labs)
    
    # Combine train and test data and labels
    combined_data = np.concatenate((train_data, test_data), axis=0)
    combined_labels = np.concatenate((train_labs, test_labs), axis=0)
    
    # Apply t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, random_state=42)
    tsne_data = tsne.fit_transform(combined_data)
    
    # Separate t-SNE data back into train and test sets
    train_tsne_data = tsne_data[:len(train_data)]
    test_tsne_data = tsne_data[len(train_data):]


    # Plot t-SNE scatterplot
    plt.figure(figsize=(5,5))
    fig, ax = plt.subplots()

    colors = ['maroon', 'lime', 'navy', 'coral', 'purple', 'gold', 'magenta', 'gray', 'darkgreen', 'teal']
    ccmap = (mpl.colors.ListedColormap(colors))
        
    scatter = plt.scatter(test_tsne_data[:, 0], test_tsne_data[:, 1], c=test_labs, cmap=ccmap, marker='.', label=data_name+' KNN')
    cbar = plt.colorbar(scatter, ticks=np.arange(10), label='Classes')
    
    
    # Set the colorbar tick labels to CIFAR10 class names
    plt.xticks([])
    plt.yticks([])
    if data_name == 'MNIST':
        class_names = mnist_classes
    elif data_name == 'CIFAR10':
        class_names = classes
    cbar.ax.set_yticklabels(class_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../graphs/'+data_name+'_KNN.png')
    plt.show()
    plt.clf()
    
    confmatrix = confusion_matrix(predicted, test_labs)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.xaxis.tick_top()
    
    cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, hue=1, reverse=True, as_cmap=True)
    if data_name == 'MNIST':
        sns.heatmap(confmatrix, annot=True, fmt=".0f", linewidths=1.5, annot_kws={'size': 8}, xticklabels=mnist_classes, yticklabels=mnist_classes, cmap=cmap)
    elif data_name == 'CIFAR10':
        sns.heatmap(confmatrix, annot=True, fmt=".0f", linewidths=1.5, annot_kws={'size': 8}, xticklabels=classes, yticklabels=classes, cmap=cmap)
    plt.yticks(rotation=0)
    ax.xaxis.set_label_position('top')
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.tight_layout()
    plt.savefig('../graphs/'+data_name+'_KNN_confusion.png')
    

def kmeans(train_data,train_labs):
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    kmeans = KMeans(n_clusters=50, n_init='auto').fit(train_data)
    
    labels = train_labs
    # cast to int 
    labels = np.array(labels)
    labels = labels.astype(int)
    
    labels = np.array(labels)
    ref_labels = utils.retrieveInfo(kmeans.labels_, labels)
    num_predicted = utils.assignPredictions(kmeans.labels_, ref_labels)
    accuracy = utils.computeAccuracy(num_predicted, labels)

    accuracy = round(accuracy * 100,2)
    print('Kmeans Accuracy', accuracy, '%')
    
    tsne = TSNE(n_components=2, random_state=42)
    embedded_data = tsne.fit_transform(train_data)

    colors = ['maroon', 'lime', 'navy', 'coral', 'purple', 'gold', 'magenta', 'gray', 'darkgreen', 'teal']
    ccmap = (mpl.colors.ListedColormap(colors))

    # Create scatterplot
    plt.figure(figsize=(5, 5))
    scatter = plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=train_labs, cmap=ccmap, marker='.', label=data_name+' Kmeans')
    cbar = plt.colorbar(scatter, ticks=np.arange(10), label='Classes')
    plt.xticks([])
    plt.yticks([])
    # Set the colorbar tick labels to CIFAR10 class names
    if data_name == 'MNIST':
        class_names = mnist_classes
    elif data_name == 'CIFAR10':
        class_names = classes
    cbar.ax.set_yticklabels(class_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../graphs/'+data_name+'_Kmeans.png')
    plt.show()
    plt.clf()

def test_knn():
    with torch.no_grad():

        train_codes = torch.tensor([])
        train_labs = torch.tensor([])
        test_codes = torch.tensor([])
        test_labs = torch.tensor([])
        
        for pics,labels in train_loader:
            codes, _ = model(pics.to(device))
            train_codes = torch.cat((train_codes, codes.flatten(1).cpu()), 0)
            train_labs = torch.cat((train_labs, labels), 0)
            if len(train_codes) > 9000:
                break
            #implement feature to display pics[0]
            
        for pics,labels in test_loader:
            codes, _ = model(pics.to(device))
            test_codes = torch.cat((test_codes, codes.flatten(1).cpu()), 0)
            test_labs = torch.cat((test_labs, labels), 0)
            if len(test_codes) > 9000:
                break
        
        print('train_codes',train_codes.shape)
        print('test_codes',test_codes.shape)
    KNN(train_codes, train_labs,test_codes, test_labs)
    kmeans(train_codes,train_labs)
    
        
test_knn()