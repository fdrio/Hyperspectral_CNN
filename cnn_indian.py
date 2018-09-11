from __future__ import print_function
import argparse 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import datasets, transforms
import os 
from IP_DataSet import IPDataSet # Data Set class
__author__ = "Francisco Diaz"
__credits__ = "KGPML"
num_epochs = 10 #subject to further change

#CNN model taken from https://github.com/KGPML/Hyperspectral
# Convolutional Neural Network for Indian Pines data set
# Implementation for Pytorch of:
# https://github.com/KGPML/Hyperspectral
# Further modifications will be added as needed 

"""
Importing Data set from path file
"""
#Path of the Indian Pines mat files (each file is a separate class)
path = os.path.join(os.getcwd(),"new_indian_pines_data")


#Data Set 
training_dataset = IPDataSet(path=path,train=True)
testing_dataset = IPDataSet(path=path,train=False)

#Data Loaders


train_loader = torch.utils.data.DataLoader(
        dataset=training_dataset,
        batch_size = 32,
        shuffle=True
        )



test_loader = torch.utils.data.DataLoader(
        dataset=testing_dataset,
        batch_size = 32,
        shuffle=True
        )


#Check for cuda availabilty
use_cuda = torch.cuda.is_available()




#Loading Data and creating Loader for the Indian Pines Data Set

# Creating a simple neural network that classifies Indian Pines classes.


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
         # Parameters for the convolutional layer:
         #    Input Channels, Output Channels, Squared Kernel Size, stride 
        self.conv1 = torch.nn.Conv2d(200,500,3,stride=1)
        self.conv2 = torch.nn.Conv2d(500,100,3,stride=1)
        # Parameters for the Affine Layer:
        #   in_features,out_features,bias
        self.fcl1 = torch.nn.Linear(100*5*5,200,bias=True)
        self.fcl2 = torch.nn.Linear(200,84,bias=True)
        self.fcl3 = torch.nn.Linear(84,16,bias=True)

        #Parameters:
        # x would be the input tensor to be forward propgated
        
    def forward(self,x):
        # MaxPool2d parameters accepts a tensor and the squared kernel size
        #stride will be the same as the kernel size by default
        x = F.relu(F.max_pool2d(self.conv1(x),2,padding=1))
        x = F.relu(F.max_pool2d(self.conv2(x),2,padding=1))
        x =  x.view(-1,100*5*5)
        x = F.relu(self.fcl1(x))
        x = F.relu(self.fcl2(x))
        x = F.log_softmax(self.fcl3(x),dim=1) # dim = dimension along which log softmax will be computed
        return x 

    def name(self):
        return "CNN"



cnn = CNN() #Creating instance of the CNN model
cnn = cnn.double()

#traning the Convolutional neural network
def train(cnn,train_loader,batch_size=32):
    if(use_cuda):
        cnn = cnn.cuda() # GPU model
    optimizer = optim.Adagrad(cnn.parameters(),lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        average_loss = 0.0
        for batch_idx, (x,target) in enumerate(train_loader):
            optimizer.zero_grad() # clean the gradient 
            if(use_cuda):
                x,target=x.cuda(),target.cuda()
            output = cnn(x)
            loss = criterion(output,target)
            loss.backward()# propagate the gradients
            optimizer.step() # update parameters
            if(batch_idx%32 ==0 or batch_idx+1==len(train_loader)):
                print ('==>>> epoch: %d, batch index: %d, train loss: %.6f'
                %(epoch,batch_idx+1 ,loss))



# Testing model accuracy
def test(cnn,train_loader,test_loader):
    total_count = 0
    correct_count = 0
    average_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    if(use_cuda):
        cnn=cnn.cuda()
    for batch_idx, (x,target) in enumerate(train_loader):
        if(use_cuda):
            x,target = x.cuda(),target.cuda()
        output = cnn(x)
        loss = criterion(output,target)
        _,pred_label = torch.max(output.data,1)
        total_count += x.data.size()[0]
        print("Predicted label: %s" %pred_label)
        print("Target output: %s" %target.data)
        correct_count += ((pred_label == target.data).sum()).item()
        
        if(batch_idx+1) % 32 == 0 or (batch_idx+1) == len(test_loader):
            print ('==>>>  batch index: %d, test loss: %.6f, acc: %.3f' %(batch_idx+1,loss, (correct_count * 1.0) / total_count))




#printing size of training and testing batches

print("Length of the training data set is: %d" %len(train_loader))
print("Length of the training data set is: %d" %len(train_loader))

#training
train(cnn,train_loader)
test(cnn,train_loader,test_loader)

torch.save(cnn.state_dict(), cnn.name())



