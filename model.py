#Defines the CNN model
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

   
    def __init__(self,num_classes):
        super(Model,self).__init__()

        #Convolution layers
        out_channels = 32

        self.conv1 = nn.Conv2d(3,32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,padding=1)

        self.pool = nn.MaxPool2d(2,2)

        #Fully connected Layers (Standard NN????)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 2)
        
        #Dropout layer
        self.dropConv = nn.Dropout2d(p=0.25)
        self.dropFc = nn.Dropout(p=.5)

    def forward(self,x):
 
        x = self.pool(F.relu(self.conv1(x))) #Convolution Layer 1
        x = self.dropConv(x) #Dropout layer 1
        x = self.pool(F.relu(self.conv2(x))) #Convolution Layer 2
        x = self.dropConv(x)  #Dropout layer 2
        x = self.pool(F.relu(self.conv3(x))) #Convolution Layer 3
        x = self.dropConv(x)  #Dropout layer 3

        #Flatten
        x = x.view(-1,128 * 4 * 4)

        #FC layers
        x = F.relu(self.fc1(x))
        x = self.dropFc(x)
        #FC Dropout
        x = self.fc2(x)
        return x

