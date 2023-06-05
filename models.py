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

        # Get the output of the backbone network
        x = self.backbone(x)
    
        output = self.arcface(x, labels)

        return output



