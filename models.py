import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from loss_layers import ArcFaceLayer

# Create a model class that combines the backbone network (Resnet18) and (optinally) the ArcFace layer
# The model class will be used for training and inference

# Create a base class called EmbeddingModel that contains the backbone network
class EmbeddingModel(nn.Module):
    def __init__(self, embedding_size):
        super(EmbeddingModel, self).__init__()
        self.embedding_size = embedding_size

        # Load a pre-trained backbone network (e.g., ResNet18)
        self.backbone = models.resnet18(weights= models.ResNet18_Weights.DEFAULT)

        # Freeze the backbone network
        for param in self.backbone.parameters():
            param.requires_grad = False


        # Replace the last fc layer of the backbone network with a one that outputs embedding_size
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, self.embedding_size)


# ArcFaceModel is a subclass of EmbeddingModel that adds an ArcFace layer
class ArcFaceModel(EmbeddingModel):
    def __init__(self, num_classes, embedding_size):
        super(ArcFaceModel, self).__init__(embedding_size)

        # Number of classes
        self.num_classes = num_classes

        # Create an new ArcFace layer
        self.arcface = ArcFaceLayer(self.embedding_size, self.num_classes)
    
    def forward(self, x, labels=None):

        # Get the output of the backbone network, so that we can use it to compute the embedding vectors
        x = self.backbone(x)

        # A cpu copy of the embedding vectors with detached gradients
        embedding_vectors = x.cpu().detach()
    
        output = self.arcface(x, labels)

        return output, embedding_vectors
    
    # Add a method to get the embedding vector
    def get_embedding(self, x):
        #no grad
        with torch.no_grad():
            return self.backbone(x)

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

