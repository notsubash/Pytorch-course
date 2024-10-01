import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self, transform=None):
        #data loading
        xy = np.loadtxt('/Users/subashpandey/Desktop/Pytorch-course/data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        # first column is the class label and the rest are features
        self.x = xy[:, 1:] # size [n_samples, n_features]
        self.y = xy[:,[0]] #n_samples, 1)
        self.n_samples = xy.shape[0]

        self.transform = transform

    # we will support indexing such that the dataset[i] can be used to get the i-th sample
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    # can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets 
    
print('Without Transform')
dataset = WineDataset()
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

print('\nWith Tensor Transform')
dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

print('\nWith Tensor and Multiplication Transform')
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)