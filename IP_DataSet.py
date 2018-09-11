from __future__ import print_function,division
import torch
import os
import scipy.io 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class IPDataSet(Dataset):
    def __init__(self,path,train,transform=None):
        """
        Args: 
            path (string): path of the Matlab file
            train (boolean): Select between training or validation data set 
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if(train):
            select ="Training"
            patch_type = "train"
        else:
            select = "Testing"
            patch_type = "testing"
        self.tensors = []

        self.labels = []
        self.transform = transform
        #iterate over every tensor class and add patch image and label to the;w
        #corresponding list
        for file in os.listdir(path):
            if(os.path.isfile(os.path.join(path,file)) and select in file):
                temp = scipy.io.loadmat(os.path.join(path,file))#loading mat dictionary
                temp = {k:v for k, v in temp.items() if k[0] != '_'} # filtering dictionary

                for i  in range(len(temp[patch_type+"_patches"])):
                    self.tensors.append(temp[patch_type+"_patches"][i])
                    self.labels.append(temp[patch_type+"_labels"][0][i])
        self.tensors = np.array(self.tensors)
        self.labels = np.array(self.labels)

    def __len__(self):
        try:
            if len(self.tensors) != len(self.labels):
                raise Exception("Lengths of the tensor and labels list are not the same") 
        except Exception as e:
            print(e.args[0])
        return len(self.tensors)


#Returns patch tensor and patch label
    def __getitem__(self,idx):
        sample = (self.tensors[idx],self.labels[idx])
       # print(self.labels)
        sample = (torch.from_numpy(self.tensors[idx]),torch.from_numpy(np.array(self.labels[idx])).long())
        return sample 
    #tuple containing the image patch and its corresponding label
