import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio

def visualize_embeddings(all_embeddings, all_labels):
    
    #Get the number of epochs
    num_epochs = len(all_embeddings['train'])

    #Num classes
    num_classes = len(np.unique(all_labels['train'][0]))

    # Create a GIF of the embeddings

    #Create list to hold frames for train and val
    train_frames = []
    val_frames = []

    #Create gif for train
    for epoch in range(num_epochs):
        train_embeddings = all_embeddings['train'][epoch]
        train_labels = all_labels['train'][epoch]

        #Create a scatter plot with colored labels
        fig = plt.figure(figsize=(10,10))
        for class_id in range(num_classes):
            class_indices = np.where(train_labels == class_id)[0]
            plt.scatter(train_embeddings[class_indices, 0], train_embeddings[class_indices, 1], label=f"Class {class_id}")
        
        plt.title(f"Epoch [{epoch+1}/{num_epochs}] - Training Embeddings")
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()

        #Save the plot as an image
        plt.savefig('data/train_epoch_{}.png'.format(epoch))
        plt.close(fig)

        #Add the image to the list of frames
        train_frames.append(imageio.imread('data/train_epoch_{}.png'.format(epoch)))
    
    #Save the list of frames as a GIF
    imageio.mimsave('data/train_embeddings.gif', train_frames, duration=500)
