import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# Create nn.module class called ArcFace that can be plugged in at the end of any backbone network

class ArcFaceLayer(nn.Module):
    def __init__(self, in_features, num_classes, s=8, m=0.50):
        super(ArcFaceLayer, self).__init__()

        #Margin parameter and scaling factor
        self.s = s
        self.m = m

        #Input feature dimension and output number of classes
        self.in_features = in_features
        self.num_classes = num_classes

        #Linear Layer with no bias
        self.fc = nn.Linear(self.in_features, self.num_classes, bias=False)

    def forward(self, x, labels):

        #Norm of weights
        w_l2 = F.normalize(self.fc.weight, p=2)

        #Norm of input features
        x_l2 = F.normalize(x, p=2)

        #Cosine similarity between input features and weights
        cos_theta = F.linear(x_l2, w_l2)

        # Clamp the values between -1 and 1 with 1e-7 for numerical stability
        cos_theta = torch.clamp(cos_theta, -1 + 1e-7, 1 - 1e-7)

        #Get the abgle using arccos
        theta = torch.acos(cos_theta)

        #Add margin to the angle
        theta += self.m

        #Apply cosine to the angle to get adjusted cosine similarity
        cos_theta = torch.cos(theta)

        #One hot encode labels
        one_hot = torch.zeros(cos_theta.size(), device='cuda')

        #Fill the one hot encoded tensor with 1s at the label indices
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        #Scale the cosine similarity by the scaling factor and multiply with one hot encoded labels
        output = cos_theta * self.s * one_hot
       
        return output


# Create a model class that combines the backbone network (Resnet18) and the ArcFace layer
# The model class will be used for training and inference

class ArcFaceModel(nn.Module):
    def __init__(self, num_classes, embedding_size):
        super(ArcFaceModel, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes

        # Load a pre-trained backbone network (e.g., ResNet18)
        self.backbone = models.resnet18(pretrained=True)

        # Get number of features of the last layer of the backbone network
        self.in_features = self.backbone.fc.in_features

        #Modify the fc layer of the backbone network to be an instance of the ArcFaceLayer layer
        self.backbone.fc = ArcFaceLayer(self.in_features, self.num_classes)
    
    def forward(self, x, labels=None):
        output = self.backbone(x, labels)
        return output



