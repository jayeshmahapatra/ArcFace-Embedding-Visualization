import torch
import torch.nn as nn
import torch.nn.functional as F

# Create nn.module class called ArcFace that can be plugged in at the end of any backbone network

class ArcFaceLayer(nn.Module):
    def __init__(self, in_features, num_classes, s=15, m=0.5):
        super(ArcFaceLayer, self).__init__()

        #Margin parameter and scaling factor
        self.s = s
        self.m = m

        #Input feature dimension and output number of classes
        self.in_features = in_features
        self.num_classes = num_classes

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels):

        #Cosine similarity between normalized input features and normalized weights
        cos_theta = F.linear(F.normalize(x), F.normalize(self.weight))

        # Clamp the values between -1 and 1 with 1e-7 for numerical stability
        cos_theta = torch.clamp(cos_theta, -1 + 1e-7, 1 - 1e-7)

        #Get the angle using arccos
        theta = torch.acos(cos_theta)

        #Add margin to the angle
        theta += self.m

        #Apply cosine to the angle to get adjusted cosine similarity
        adjusted_cos_theta = torch.cos(theta)

        #One hot encode labels
        one_hot = torch.zeros(cos_theta.size(), device=x.device)

        #Fill the one hot encoded tensor with 1s at the label indices
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = (one_hot * adjusted_cos_theta) + ((1.0 - one_hot) * cos_theta)
        output *= self.s
       
        return output