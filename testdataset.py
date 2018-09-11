import os
from IP_DataSet import IPDataSet
import numpy as np
import torch
import random as rand
from torchvision import transforms, utils


path = "/Users/francisco/Desktop/Summer/Pytorch/new_indian_pines_data"
trans = transforms.ToTensor()
training_dataset = IPDataSet(path=path,train=True,transform=trans)
testing_dataset = IPDataSet(path=path,train=False,transform=trans)
print("Sampling from Data Set:")
print("Sample from training set at %d ith index: %s" %(100,training_dataset.__getitem__(100)[0].size()))
print("Length of the training tensor: %d" %training_dataset.__len__())

print("Length of the testing tensor: %d" %testing_dataset.__len__())

#print("Type Tensor: %s" %training_dataset.__getitem__(100)[0].data)
print("Type Target: %s" %type(training_dataset.__getitem__(100)[1].item()))

