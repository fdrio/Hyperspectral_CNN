from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import os
import numpy as np
root = './data'
if not os.path.exists(root): # if path does not exist => create path
    os.mkdir(root)

batch_size =32



#transformations applied to the data
trans = transforms.Compose([transforms.ToTensor()])

#training set with their respective
train_set = datasets.MNIST(root=root,train=True,transform =trans,download=True)
test_set = datasets.MNIST(root=root,train=False,transform =trans,download=True)

#train loader
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
#test loader
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=True)

print ('==>>> total trainning batch number: %d' %len(train_loader))
print ('==>>> total testing batch number: %d'%len(test_loader))


print(train_set.__getitem__(1))

for batch_idx, (x,target) in enumerate(train_loader):
                print("Input  %s" %x)
                print("Target  %s" %target)


