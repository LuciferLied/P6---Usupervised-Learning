import torch
from tqdm import tqdm
from util import Models as Model
from torchvision.datasets import CIFAR10, MNIST
from util import CSV as csvUtil
from Test_model_toCSV import test
from torchvision import transforms
import torch.utils.data as data

# set device
if torch.cuda.is_available():
    #print('Using GPU')
    dtype = torch.float32
    device = torch.device('cuda')
else:
    print('Using CPU')
    device = torch.device('cpu')

def load_data(data_set, batch_size):
    
    class AUG_PAIR(data_set):
        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]

            if self.transform is not None:
                pos_1 = self.transform(img)
                pos_2 = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return pos_1, pos_2, target

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    train_data = AUG_PAIR(root='data', train=True, transform=train_transform, download=True)
    aug_train = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True,drop_last=True)
    
    test_train_data = data_set(root='data', train=True, download=True, transform=test_transform)
    aug_test_train = data.DataLoader(test_train_data, batch_size=batch_size, shuffle=True)
    
    test_test_data = data_set(root='data', train=False, transform=test_transform, download=True)
    aug_test_test = data.DataLoader(test_test_data, batch_size=batch_size, shuffle=True)
    
    return aug_train, aug_test_train, aug_test_test

def train(data_name,aug_train, aug_test_train, aug_test_test, epochs, batch_size, lr, load, neighbors_cluster):
    
    if load == True:
        load_model = 'trained_models/chk/Res18_CIFAR10_30_0.001.pth'
        model = torch.load(load_model, map_location=torch.device(device))
        temp = load_model.split('/')
        vars = temp[1].split('_')
        name = vars[0]
        data_name = vars[1]
        pretrained_epochs = vars[2]
        pretrained_epochs = int(pretrained_epochs)
        print('Model:', name,'pretrained with:', data_name, 'epochs:', pretrained_epochs, 'lr:', lr)
        lr = vars[3].replace('.pth', '')
        lr = float(lr)
    else:
        model = Model.simCLR(data_name)
        pretrained_epochs = 0

    model.to(device)
    name = model.__class__.__name__
    
    #Format print
    print('Training {} on {}, with {} epochs and batch size: {} lr {}'.format(name, data_name, epochs, batch_size, lr))

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    temperature = 0.5

    def contrastive_loss(out_1, out_2, temperature):
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

        return loss

    # Train
    for epoch in range(epochs):
        total_loss, total_num, train_bar = 0.0, 0, tqdm(aug_train)
        for pics1, pics2, labels in train_bar:
            pics1 = pics1.to(device)
            pics2 = pics2.to(device)

            # Forward
            _, out_1 = model(pics1)
            _, out_2 = model(pics2)
            
            loss = contrastive_loss(out_1, out_2, temperature)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_num += batch_size
            total_loss += loss.item() * batch_size
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch + pretrained_epochs + 1, epochs + pretrained_epochs, total_loss / total_num))
        #save model for loading later
        if ((epoch + 1) % 5 == 0) or (epoch == epochs - 1):
            print('Saving model as: ', 'trained_models/chk{}_{}_{}_{}_{}.pth'.format(name, data_name, epoch + 1 + pretrained_epochs, batch_size, lr))
            saveAsChkPnt = 'trained_models/chk/{}_{}_{}_{}_{}.pth'.format(name, data_name, epoch + 1 + pretrained_epochs, batch_size, lr)
            torch.save(model.state_dict(), saveAsChkPnt)
        
        if ((epoch + 1) % 5 == 0) or (epoch == epochs - 1):
            knn_accuracy, kmeans_accuracy = test(model, aug_test_train, aug_test_test, device, neighbors_cluster)
            values = [epoch+1+pretrained_epochs, batch_size, lr, loss.item(), knn_accuracy, kmeans_accuracy]
            csvUtil.saveToCSV(values)

    torch.save(model, 'trained_models/{}_{}_{}_{}_{}.pth'.format(name, data_name, epoch + 1 + pretrained_epochs, batch_size, lr))

#Settings
total_epochs = 30
batch_sizes = [256, 512, 1024, 2048]
learns = [0.001]
neighbors_cluster = 20


data_set = CIFAR10
data_name = data_set.__name__

load = False

for size in batch_sizes:
    aug_train, aug_test_train, aug_test_test = load_data(data_set, size)
    for lr in learns:
        train(data_name ,aug_train, aug_test_train, aug_test_test, total_epochs, size, lr, load, neighbors_cluster)
