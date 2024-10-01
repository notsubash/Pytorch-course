#Divide the sample into small batches

#Looks like this:

# #training loop
# for epoch in range(1000):
#     #loop over all batches
#     for i in range(total_batches):
#         x_batch, y_batch = ...


#Terminology to remember
'''
epoch = 1 forward and backward pass of ALL training samples

batch_size = number of training samples in one forward & backward pass

number of iterations = number of passes, each pass using [batch_size] number of samples

e.g., 100 samples, batch_size=20 --> 100/20 = 5 iterations for 1 epoch
'''

# --> use DataSet and DataLoader to load wine.csv
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# Implement a custom Dataset
# inherit Dataset
# implementing __init__ , __getitem__ , and __len__

class WineDataset(Dataset):
    def __init__(self):
        #data loading
        xy = np.loadtxt('/data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        # first column is the class label and the rest are features
        self.x = torch.from_numpy(xy[:, 1:]) # size [n_samples, n_features]
        self.y = torch.from_numpy(xy[:,[0]]) #n_samples, 1)
        self.n_samples = xy.shape[0]

    # we will support indexing such that the dataset[i] can be used to get the i-th sample
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    # can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
dataset = WineDataset()

# Checking the first sample 
# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)


dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

dataiter = iter(dataloader)
data = next(dataiter)
features, labels =data
print(features, labels)

#dummy training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        #here: 178 samples, batch_size = 4, n_iters = 178/4 = 44.5 --> 45 iterations
        if (i+1) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')

#some famous datasets available in torchvision.datasets
# fashion-mnist, cifar, coco

#MNIST Dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train =True, transform=torchvision.transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=3, shuffle=True)

#looking at one random sample
dataiter = iter(train_loader)
data = next(dataiter)
inputs, targets = data
print(inputs.shape, targets.shape)

