# https://clay-atlas.com/us/blog/2021/08/03/machine-learning-en-introduction-autoencoder-with-pytorch/

import csv
import os
import math
import glob
import shutil
import torch
from util import Models as Model
from util import CSV as csvUtil
from Test_model_toCSV import test



def train(epochs, batch_size, lr, load):

    import torch
    import torch.nn as nn
    import torch.utils.data as data
    from torchvision import datasets
    from torchvision.transforms import ToTensor
    import time 


    start = time.time()
    # set device
    if torch.cuda.is_available():
        #print('Using GPU')
        dtype = torch.float32
        device = torch.device('cuda:0')
    else:
        print('Using CPU')
        device = torch.device('cpu')



    # DataLoader
    train_set = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    
    # DataLoader
    test_set = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Optimizer and loss function

    model_name = str(model)
    model_name = model_name.split('(')
    #print(model_name[0])

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()
    
    lastEpoch = 0
    
    if load == True:
        model.load_state_dict(torch.load('models/modelChkPnt.pth'))
        model.eval()
        print("Model loaded")
        

        lastLine = csvUtil.lastLineCSV()
        lastLine = lastLine.split(',')
        lastEpoch = int(lastLine[0])
        if lastEpoch != total_epochs:
            epochs = epochs - lastEpoch
        

    # Train
    for epoch in range(epochs):
        start = time.time()
        for data, labels in train_loader:

            inputs = data.view(-1, 1, 784)
            inputs = inputs.to(device)
            
            # Forward
            codes, decoded = model(inputs)

            # Backward
            optimizer.zero_grad()
                    
            loss = loss_function(decoded, inputs)
            loss.backward()
            optimizer.step()
        
        #save model for loading later
        saveAsChkPnt = 'models/modelChkPnt.pth'
        saveAs = 'models/model.pth'
        torch.save(model.state_dict(), saveAsChkPnt)
        torch.save(model, saveAs)
        accuracy = test(saveAs, test_set, device)
        
        values = [epoch+1+lastEpoch, batch_size, lr, loss.item(), accuracy, time.time() - start]
        #print(values)
        
        csvUtil.saveToCSV(values)
        
        #print("Model saved as: ", saveAs)
        #print(loss.item())


        # Show progress
        #print('[{}/{}] Loss:'.format(epoch+1, epochs), loss.item())

    #print(codes.shape)

    #print('Finished Training using', device)
    #print('Time: ', time.time() - start)         

filePaths = []
filePaths = glob.glob('models/*.pth')
load = False

#Settings
model = Model.Smol_AutoEncoder()
total_epochs = 20
batch_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
lr_start = 0.001
lr_limit = 0.005 #Skal være højere end reel limit
lr_increment = 0.001
lr = lr_start

if filePaths.__len__() == 0:
    load = False
    csvUtil.delCSVContent()
    
    for size in batch_sizes:
        if lr == lr_limit:
            lr = lr_start
        while lr < lr_limit:
            lr = round(lr, 3)
            #print('Batch Size: ', size, 'Learning Rate: ', lr)
            train(total_epochs, size, lr, load)
            lr += lr_increment
    
elif filePaths.__len__() != 0:  
    load = True
    lastLine = csvUtil.lastLineCSV()
    lastLine = lastLine.split(',')
    for i in range(0, lastLine.__len__()):
        lastLine[i] = float(lastLine[i])

    foo = batch_sizes
    
    if int(lastLine[0]) == total_epochs:
        if lr_limit-lr_increment == lastLine[2]:
            while int(lastLine[1]) in foo:
                batch_sizes.pop(0)
                foo = batch_sizes
        elif lr_limit-lr_increment != lastLine[2]:
            for i in range(0, foo.__len__()):
                if foo[0] != lastLine[1]:
                    batch_sizes.pop(0)
                    foo = batch_sizes
        lr = lastLine[2] + lr_increment
        load = False
            
    elif int(lastLine[0]) != total_epochs:
        if lr_limit-lr_increment != lastLine[2]:
            for i in range(0, foo.__len__()):
                if foo[0] != lastLine[1]:
                    batch_sizes.pop(0)
                    foo = batch_sizes
        lr = lastLine[2]

    for size in foo:
        if lr == lr_limit:
            lr = lr_start
        while lr < lr_limit:
            lr = round(lr, 3)
            print('Batch Size: ', size, 'Learning Rate: ', lr)
            train(total_epochs, size, lr, load)
            lr += lr_increment
