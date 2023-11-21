# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
import torchvision.transforms
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt
import numpy

my_transform = torchvision.transforms.ToTensor()
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform= my_transform)
val_dataset = dsets.MNIST(root='./data', train=False, download=True, transform= my_transform)

# Data visualization
def show_data(data_sample):
    plt.imshow(data_sample[0].numpy(), cmap='gray')
    plt.title('y = '+ str(data_sample[1]))
    print(np.shape(data_sample))

show_data(train_dataset[3])

class CNN(nn.Module):

    # Contructor
    def __init__(self, Layers, Kernels, MaxPoolKernels, Padding):
        super(CNN, self).__init__()
        self.hidden = nn.ModuleList()
        self.maxpool = nn.ModuleList()
        idx = 0
        for input_size, output_size in zip(Layers, Layers[1:]):
            self.hidden.append(nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=Kernels[idx], padding= Padding[idx]))
            self.maxpool.append(nn.MaxPool2d(kernel_size=MaxPoolKernels[idx]))
            idx+=1

    # Prediction
    def forward(self, activation):
        L = len(self.hidden)
        for (l, cnn) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = nn.relu(cnn(activation))
                activation = self.maxpool[l](activation)
            else:
                activation = nn.relu(cnn(activation))
                activation = self.maxpool[l](activation)
                activation = activation.view(activation.size(0), -1)
        return activation

    # Outputs in each steps
    def activations(self, x):
        L = len(self.hidden)
        activations = []
        for (l, cnn) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = nn.relu(cnn(activation))
                activation = self.maxpool[l](activation)
            else:
                activation = nn.relu(cnn(activation))
                activation = self.maxpool[l](activation)
                activation = activation.view(activation.size(0), -1)
            activations.append(activation)
        return activations


class Net(nn.Module):

    # Constructor
    def __init__(self, Layers):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(Layers, Layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))

    # Prediction
    def forward(self, activation):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = nn.relu(linear_transform(activation))
            else:
                activation = linear_transform(activation)
        return activation


class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x1)
        return x2

# CNN Model hyperparameters
Layers = []