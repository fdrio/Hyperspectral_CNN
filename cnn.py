from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os

# CNN class

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 5x5 kernel
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d() # Dropout ensemble
        self.fc1 = nn.Linear(320, 50) # Affine Layer
        self.fc2 = nn.Linear(50, 10) # Affine Layer



#forward method
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320) # convert into a 1x320 row vector (vectorize the tensor)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=True)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


    def name(self):
        return "CNN"


## load mnist dataset
use_cuda = torch.cuda.is_available() # check for cuda

root = './data'
if not os.path.exists(root): # if path does not exist => create path
    os.mkdir(root)
batch_size=32




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



 #training procedure
cnn = CNN() #our model

def train(train_loader, test_loader, batch_size=32):
    if(use_cuda):# if cuda is available => use cuda
    #Choose optimizer
        cnn.cuda()
    optimizer = optim.SGD(cnn.parameters(),lr=0.01,momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10): # epoch size (chosen randomly)
        average_loss = 0.0
        for batch_idx, (x,target) in enumerate(train_loader):
            optimizer.zero_grad() # Initialize the gradient to 0.0 for every training iteration
            if(use_cuda):
                x,target = x.cuda(),target.cuda() # CUDA
            output = cnn(x) # forward
            loss = criterion(output,target) #  Cross Entropy Loss (output,target)
            loss.backward() # Computes the gradients for all leaf nodes
            optimizer.step() # Update parameters

            if(batch_idx%32 ==0 or batch_idx+1==len(train_loader)):
                print ('==>>> epoch: %d, batch index: %d, train loss: %.6f'
                %(epoch,batch_idx+1 ,loss))

def test(train_loader,test_loader):
    total_count = 0
    correct_count = 0
    ave_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    if(use_cuda):
        cnn.cuda()
    for batch_idx,(x,target) in enumerate(test_loader):
        if(use_cuda):
            x,target = x.cuda(),target.cuda()
        output = cnn(x)

        loss = criterion(output,target)

        # torch.max returns a tuple of the vector with largest components in each row vector and a vector with the index of that largest component
        _, pred_label = torch.max(output.data, 1)
        total_count += x.data.size()[0] # the number of rows of the matrix give us the the batch size
        print("Predicted label: %s" %pred_label)
        print("Target output: %s" %target.data)
        correct_count+=((pred_label==target.data).sum()).item()

        #output
        if(batch_idx+1) % 32 == 0 or (batch_idx+1) == len(test_loader):
            print ('==>>>  batch index: %d, test loss: %.6f, acc: %.3f' %(batch_idx+1,loss, (correct_count * 1.0) / total_count))


#training
train(train_loader,test_loader)
test(train_loader,test_loader)

torch.save(cnn.state_dict(), cnn.name())





















