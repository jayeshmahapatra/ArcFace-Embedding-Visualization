import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from models import FaceModel

# Hyperparameters
num_epochs = 10
batch_size = 64
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to train the model
def train(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    best_val_loss = float("inf")  # Initialize with a very high value
    model.train()

    train_loop = tqdm(total=len(train_loader), leave=False)
    train_loop.set_description(f"Epochs: 0/{num_epochs}")


    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data, target)  # Pass both data and target to the model's forward method
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            train_loop.update(1)

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for val_data, val_target in tqdm(val_loader, leave=False):
                val_data, val_target = val_data.to(device), val_target.to(device)
                val_output = model(val_data, val_target)  # Pass both val_data and val_target to the model's forward method
                val_loss += criterion(val_output, val_target).item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/best_model.pth")  # Save the model with the best validation loss

        train_loop.set_description(f"Epochs: {epoch+1}/{num_epochs}")
        train_loop.set_postfix(loss=loss.item(), avg_loss=avg_loss, val_loss=val_loss)

        train_loop.reset()

    train_loop.close()


    print("Training complete!")


# __main__ function
if __name__ == "__main__":

    # Data processing and augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load the CIFAR10 dataset
    dataset = CIFAR10(root="./data/cifar10", train=True, transform=transform, download=True)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create an instance of FaceModel
    model = FaceModel(num_classes=10, embedding_size=2, use_arcface=True).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model using train() function
    train(model, train_loader, val_loader, optimizer, criterion, num_epochs)
