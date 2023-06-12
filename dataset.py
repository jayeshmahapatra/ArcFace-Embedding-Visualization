import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import pandas as pd



class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, annotations_file, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotations_file, sep=" ", header=None)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        label = torch.tensor(self.annotations.iloc[index, 1])
        
        if self.transform:
            image = self.transform(image)
            
        return (image, label)