from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import statistics

def readFromCSV(file):
    
    rootdir = os.getcwd()

    for subdir, dirs, files in os.walk(rootdir):
        
        filepath = subdir + os.sep

        if filepath.__contains__("dataanalysis"):
            filepath = filepath + file
            with open(filepath, 'r', ) as CSVstats:
                reader = csv.reader(CSVstats)
                data = list(reader)
                return data

data = readFromCSV("csvstats_old.csv")
data_holder = []

print()

#sort data by epochs, then by batch size, then by learning rate



print(data_holder)
    

for index in data:
        data[data.index(index)] = [float(index[0]), float(index[1]), float(index[2]), float(index[3]), float(index[4]), float(index[5])]

epochs = []
batch_size = []
learning_rate = []
loss = []
acuarracy = [] 

for i in data:
        epochs.append(i[0]) #"Epochs"
        batch_size.append(i[1]) #"Batch Size"
        learning_rate.append(i[2]) #"Learning Rate"
        loss.append(i[3]) #"Loss"
        acuarracy.append(i[4]) #"Accuracy"
        
ep_sep = []
batch_size_sep = []
lr_sep = []
mean_acc_batch = []
mean_time_batch = []

#Different values used for epochs, batch size and learning rate
for i in range(len(epochs)):
    if epochs[i] not in ep_sep:
        ep_sep.append(epochs[i])
    if batch_size[i] not in batch_size_sep:
        batch_size_sep.append(batch_size[i])
    if learning_rate[i] not in lr_sep:
        lr_sep.append(learning_rate[i])

ep_sep = sorted(ep_sep)
batch_size_sep = sorted(batch_size_sep)
lr_sep = sorted(lr_sep)

    
def deep_index(lst, epoch, batch_size, lr):
    epoch = [(i, sub.index(epoch)) for (i, sub) in enumerate(lst) if epoch in sub]
    batch_size = [(i, sub.index(batch_size)) for (i, sub) in enumerate(lst) if batch_size in sub]
    lr = [(i, sub.index(lr)) for (i, sub) in enumerate(lst) if lr in sub]

    for i in range(epoch.__len__()):
        epoch[i] = epoch[i][0]
    for i in range(batch_size.__len__()):
        batch_size[i] = batch_size[i][0]
    for i in range(lr.__len__()):
        lr[i] = lr[i][0]

    holder = (set(epoch).intersection(batch_size, lr))
    
    return holder

print(deep_index(data, 1, 8, 0.001))




#mean accuracy and time for each batch size        
for i in range(len(batch_size_sep)):
    holder = []
    time_holder = []
    for j in range(len(data)):
        if data[j][1] == batch_size_sep[i]:
            holder.append(data[j][4])
            time_holder.append(data[j][5])
    mean_acc_batch.append(statistics.mean(holder))
    mean_time_batch.append(statistics.mean(time_holder))
    
#mean accuracy for learning rate
mean_acc_lr = []
for i in range(len(lr_sep)):
    holder = []
    for j in range(len(data)):
        if data[j][2] == lr_sep[i]:
            holder.append(data[j][4])
    mean_acc_lr.append(statistics.mean(holder))
    
#highest accuracy for each for each learning rate
highest_acc_lr = []
for i in range(len(lr_sep)):
    holder = []
    for j in range(len(data)):
        if data[j][2] == lr_sep[i]:
            holder.append(data[j][4])
    highest_acc_lr.append(max(holder))
    
#highest and mean accuracy for each epoch
highest_acc_ep = []
mean_acc_ep = []
for i in range(len(ep_sep)):
    holder = []
    for j in range(len(data)):
        if data[j][0] == ep_sep[i]:
            holder.append(data[j][4])
    highest_acc_ep.append(max(holder))
    mean_acc_ep.append(statistics.mean(holder))
    
#highest and mean accuracy for each batch size
highest_acc_batch = []
mean_acc_batch = []
for i in range(len(batch_size_sep)):
    holder = []
    for j in range(len(data)):
        if data[j][1] == batch_size_sep[i]:
            holder.append(data[j][4])
    highest_acc_batch.append(max(holder))
    mean_acc_batch.append(statistics.mean(holder))

#Plotting the data
figure, axis = plt.subplots(5, 4, figsize=(20, 20))

#row 1
axis[0, 0].plot(acuarracy, loss, 'ro')
axis[0, 0].set_xlabel('Accuracy')
axis[0, 0].set_ylabel('Loss')

axis[0, 1].plot(acuarracy, epochs, 'bo')
axis[0, 1].set_xlabel('Accuracy')
axis[0, 1].set_ylabel('Epochs')

axis[0, 2].plot(acuarracy, batch_size, 'go')
axis[0, 2].set_xlabel('Accuracy')
axis[0, 2].set_ylabel('Batch Size')

axis[0, 3].plot(acuarracy, learning_rate, 'yo')
axis[0, 3].set_xlabel('Accuracy')
axis[0, 3].set_ylabel('Learning Rate')

#row 2
axis[1, 0].plot(loss, acuarracy, 'ro')
axis[1, 0].set_xlabel('Loss')
axis[1, 0].set_ylabel('Accuracy')

axis[1, 1].plot(epochs, acuarracy, 'bo')
axis[1, 1].set_xlabel('Epochs')
axis[1, 1].set_ylabel('Accuracy')

axis[1, 2].plot(batch_size, acuarracy, 'go')
axis[1, 2].set_xlabel('Batch Size')
axis[1, 2].set_ylabel('Accuracy')

axis[1, 3].plot(learning_rate, acuarracy, 'yo')
axis[1, 3].set_xlabel('Learning Rate')
axis[1, 3].set_ylabel('Accuracy')

#row 3
axis[2, 1].plot(ep_sep, mean_acc_ep, 'bo')
axis[2, 1].plot(ep_sep, mean_acc_ep)
axis[2, 1].set_xlabel('Epochs')
axis[2, 1].set_ylabel('Mean Accuracy')

axis[2, 2].plot(batch_size_sep, mean_acc_batch, 'go')
axis[2, 2].plot(batch_size_sep, mean_acc_batch)
axis[2, 2].set_xlabel('Batch Size')
axis[2, 2].set_ylabel('Mean Accuracy')

axis[2, 3].plot(lr_sep, mean_acc_lr, 'yo')
axis[2, 3].plot(lr_sep, mean_acc_lr)
axis[2, 3].set_xlabel('Learning Rate')
axis[2, 3].set_ylabel('Mean Accuracy')

#row 4
axis[3, 1].plot(ep_sep, highest_acc_ep, 'bo')
axis[3, 1].plot(ep_sep, highest_acc_ep)
axis[3, 1].set_xlabel('Epochs')
axis[3, 1].set_ylabel('Highest Accuracy')

axis[3, 2].plot(batch_size_sep, highest_acc_batch, 'go')
axis[3, 2].plot(batch_size_sep, highest_acc_batch)
axis[3, 2].set_xlabel('Batch Size')
axis[3, 2].set_ylabel('Highest Accuracy')

axis[3, 3].plot(lr_sep, highest_acc_lr, 'yo')
axis[3, 3].plot(lr_sep, highest_acc_lr)
axis[3, 3].set_xlabel('Learning Rate')
axis[3, 3].set_ylabel('Highest Accuracy')

#row 5
axis[4, 2].plot(batch_size_sep, mean_time_batch, 'go')
axis[4, 2].plot(batch_size_sep, mean_time_batch)
axis[4, 2].set_xlabel('Batch Size')
axis[4, 2].set_ylabel('Mean Time (s/epoch)')

plt.show()