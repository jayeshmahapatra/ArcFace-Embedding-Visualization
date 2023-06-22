import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from loss_layers import ArcFaceLayer

# Create resusable VGGBlock to be used to create a VGG8 network

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(VGGBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    
# Create a VGG8 network with Softmax layer at the end

class VGG8Softmax(nn.Module):
    def __init__(self, num_features, num_classes):
        super(VGG8Softmax, self).__init__()
        self.block1 = VGGBlock(1, 16, 2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = VGGBlock(16, 32, 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3 = VGGBlock(32, 64, 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorm = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 3 * 3, num_features)
        self.batchnorm2 = nn.BatchNorm1d(num_features)
        self.fc2 = nn.Linear(num_features, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, labels=None):
        x = self.block1(x)
        x = self.maxpool1(x)
        x = self.block2(x)
        x = self.maxpool2(x)
        x = self.block3(x)
        x = self.maxpool3(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.batchnorm2(x)
        embedding_vectors = x.cpu().detach()
        x = self.fc2(x)
        return x, embedding_vectors
    
    def get_embedding(self, x):
            
            #no grad
            with torch.no_grad():
                x = self.block1(x)
                x = self.maxpool1(x)
                x = self.block2(x)
                x = self.maxpool2(x)
                x = self.block3(x)
                x = self.maxpool3(x)
                x = self.batchnorm(x)
                x = self.dropout(x)
                x = self.flatten(x)
                x = self.fc1(x)
                x = self.batchnorm2(x)
                return x

    
# Create a VGG8 network with ArcFace layer at the end
    
class VGG8ArcFace(nn.Module):
    def __init__(self, num_features, num_classes):
        super(VGG8ArcFace, self).__init__()
        self.block1 = VGGBlock(1, 16, 2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = VGGBlock(16, 32, 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3 = VGGBlock(32, 64, 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorm = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 3 * 3, num_features)
        self.batchnorm2 = nn.BatchNorm1d(num_features)
        self.arcface = ArcFaceLayer(num_features, num_classes)

    # Forward pass, returns the embedding vectors and the output of the ArcFace layer
    def forward(self, x, labels):
        x = self.block1(x)
        x = self.maxpool1(x)
        x = self.block2(x)
        x = self.maxpool2(x)
        x = self.block3(x)
        x = self.maxpool3(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.batchnorm2(x)
        embedding_vectors = x.cpu().detach()
        output = self.arcface(x, labels)

        return output, embedding_vectors

    def get_embedding(self, x):

        #no grad
        with torch.no_grad():
            x = self.block1(x)
            x = self.maxpool1(x)
            x = self.block2(x)
            x = self.maxpool2(x)
            x = self.block3(x)
            x = self.maxpool3(x)
            x = self.batchnorm(x)
            x = self.dropout(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.batchnorm2(x)
            return x

