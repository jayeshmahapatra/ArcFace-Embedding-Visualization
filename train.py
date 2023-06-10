import torch
import random
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import MNIST
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
import os
import pandas as pd

from models import ArcFaceModel
from visualization import visualize_embeddings
from dataset import CelebADataset


# Hyperparameters
num_epochs = 80
batch_size = 4
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the random seed for reproducible results
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# Function to train the model
def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, save_embeddings=False, visualize_val=False):
    best_val_loss = float("inf")  # Initialize with a very high value
    model.train()

    train_loop = tqdm(total=len(train_loader), leave=False)
    train_loop.set_description(f"Epochs: 0/{num_epochs}")

    table_data = []  # Table data to store epoch, train loss, and val loss

    #Dictionary to store the embeddings and labels for visualization
    all_embeddings = {}
    all_labels = {}

    all_embeddings['train'] = []
    all_embeddings['val'] = []

    all_labels['train'] = []
    all_labels['val'] = []

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output, train_batch_embeddings = model(data, target)  # Pass both data and target to the model's forward method

            # if epoch == 2:
            #     print(train_batch_embeddings)
            
            #Save the embeddings and labels for visualization
            if save_embeddings:
                if batch_idx == 0:
                    train_embeddings = train_batch_embeddings
                    train_labels = target.cpu().detach()
                else:
                    train_embeddings = torch.cat((train_embeddings, train_batch_embeddings), dim=0)
                    train_labels = torch.cat((train_labels, target.cpu().detach()), dim=0)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            train_loop.update(1)

        train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_idx, (val_data, val_target) in enumerate(tqdm(val_loader, leave=False)):
                val_data, val_target = val_data.to(device), val_target.to(device)
                val_output, val_batch_embeddings = model(val_data, val_target)  # Pass both val_data and val_target to the model's forward method

                if save_embeddings:
                    if batch_idx == 0:
                        val_embeddings = val_batch_embeddings
                        val_labels = val_target.cpu().detach()
                    else:
                        val_embeddings = torch.cat((val_embeddings, val_batch_embeddings), dim=0)
                        val_labels = torch.cat((val_labels, val_target.cpu().detach()), dim=0)

                val_loss += criterion(val_output, val_target).item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/best_model.pth")  # Save the model with the best validation loss

        #Save the embeddings and labels for visualization
        if save_embeddings:
            all_embeddings['train'].append(train_embeddings.numpy())
            all_embeddings['val'].append(val_embeddings.numpy())

            all_labels['train'].append(train_labels.numpy())
            all_labels['val'].append(val_labels.numpy())

        train_loop.set_description(f"Epochs: {epoch+1}/{num_epochs}")
        train_loop.set_postfix(train_loss=train_loss, val_loss=val_loss)

        train_loop.reset()

        # Store epoch, train loss, and val loss in table data
        table_data.append([epoch+1, train_loss, val_loss])

        # Create a table with epoch, train loss, and val loss
        table = tabulate(table_data, headers=["Epoch", "Train Loss", "Val Loss"], tablefmt="presto")

        # Print the table with updated data
        if epoch == 0:
            # First epoch, print the table normally
            print(table)
        else:
            # Subsequent epochs, use carriage return to overwrite the previous table
            table = tabulate(table_data, headers=["Epoch", "Train Loss", "Val Loss"], tablefmt="presto")
            print(table, end="\r")

    train_loop.close()

    print("Training complete!")

    #Create a visualization of the embeddings
    if save_embeddings:
        visualize_embeddings(all_embeddings, all_labels, visualize_val= visualize_val)
    


# __main__ function
if __name__ == "__main__":

    # Data processing and augmentation
    # Define transformations
    transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    # Create an instance of the dataset
    dataset = CelebADataset(root_dir="data/img_align_celeba", annotations_file="data/identity_CelebA.txt", transform=transform)

    #Define the number of classes. We extract the top num_classes identities from the dataset, sorted by the number of images they have
    num_classes = 6

    # Get the indices of the images belonging to the first num_classes identities
    indices = np.where(np.isin(dataset.annotations[1], np.array(dataset.annotations[1].value_counts()[:num_classes].index)))[0]

    # Create a subset of the dataset with the indices
    subset = Subset(dataset, indices)

    #Relabel the labels to be from 0 to num_classes-1
    # Create a unique mapping from the original labels to the new labels
    mapping = {k: v for v, k in enumerate(np.array(dataset.annotations[1].value_counts()[:num_classes].index))}



    # Change the labels in the subset
    for i in range(len(subset)):
        subset.dataset.annotations.iloc[subset.indices[i], 1] = mapping[subset.dataset.annotations.iloc[subset.indices[i], 1]]


    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(subset))
    val_size = len(subset) - train_size
    train_dataset, val_dataset = random_split(subset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create an instance of ArcFaceModel
    model = ArcFaceModel(num_classes=num_classes, embedding_size=2).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model using train() function
    train(model, train_loader, val_loader, optimizer, criterion, num_epochs, save_embeddings=True, visualize_val=False)
