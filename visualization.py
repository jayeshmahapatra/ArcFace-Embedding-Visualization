import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image
import imageio



# Function to visualize the embeddings using matplotlib
def visualize_embeddings(all_embeddings, all_labels, visualize_val=False):
    
    #Get the number of epochs
    num_epochs = len(all_embeddings['train'])

    #Num classes
    num_classes = len(np.unique(all_labels['train'][0]))

    # Create a GIF of the embeddings

    #Create list to hold frames for train and val
    train_frames = []
    val_frames = []

    #Get the min and max values for the train embeddings across all epochs
    x_min = np.min([np.min(all_embeddings['train'][epoch][:,0]) for epoch in range(num_epochs)])
    x_max = np.max([np.max(all_embeddings['train'][epoch][:,0]) for epoch in range(num_epochs)])
    y_min = np.min([np.min(all_embeddings['train'][epoch][:,1]) for epoch in range(num_epochs)])
    y_max = np.max([np.max(all_embeddings['train'][epoch][:,1]) for epoch in range(num_epochs)])

    #max absolute value
    max_abs = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))

    # Set the limits of the axes
    #xlim = [x_min - 0.1, x_max + 0.1]
    #ylim = [y_min - 0.1, y_max + 0.1]
    xlim = [-max_abs - 0.1, max_abs + 0.1]
    ylim = [-max_abs - 0.1, max_abs + 0.1]

    #Create gif for train
    for epoch in range(num_epochs):
        train_embeddings = all_embeddings['train'][epoch]
        train_labels = all_labels['train'][epoch]

        #Create a scatter plot with colored labels
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Define a custom colormap excluding the color red
        colors = ['blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
        cmap = ListedColormap(colors)

        # Add a unit circle to the plot
        circle = plt.Circle((0,0), 1, color='red', fill=False)
        ax.add_artist(circle)

        for class_id in range(num_classes):
            class_indices = np.where(train_labels == class_id)[0]
            ax.scatter(train_embeddings[class_indices, 0], train_embeddings[class_indices, 1], label=f"Class {class_id}", alpha=1)

            

            # Add axes to the plot with arrows
            ax.arrow(xlim[0], 0, xlim[1] - xlim[0], 0, length_includes_head=True, head_width=0.05, color='black')
            ax.arrow(0, ylim[0], 0, ylim[1] - ylim[0], length_includes_head=True, head_width=0.05, color='black')

        
        plt.title(f"Epoch [{epoch+1}/{num_epochs}] - Training Embeddings")
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        #Set the limits of the axes
        fig.gca().set_xlim([xlim[0]-0.1, xlim[1]+0.1])
        fig.gca().set_ylim([ylim[0]-0.1, ylim[1]+0.1])

        # Add the unit circle to the legend
        ax.scatter([], [], color='red', label="Unit Circle")
        
        plt.legend()

        #Save the plot as an image
        plt.savefig('data/train_epoch_{}.png'.format(epoch))
        plt.close(fig)

        #Add the image to the list of frames
        train_frames.append(imageio.imread('data/train_epoch_{}.png'.format(epoch)))
    
    #Save the list of frames as a GIF
    imageio.mimsave('data/train_embeddings.gif', train_frames, duration=125)
