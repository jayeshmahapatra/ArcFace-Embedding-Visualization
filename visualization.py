import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio

# # Import manim class
# from manim_animation import EmbeddingsAnimation

# # Function to visualize the embeddings using mamin
# def visualize_embeddings_manim(all_embeddings, all_labels, visualize_val=False):
#     animation = EmbeddingsAnimation(all_embeddings, all_labels)
#     animation.render()
#     #animation.save_as_gif(filename='data/train_embeddings_manim.gif')
#     pass



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

    # Set the limits of the axes
    xlim = [x_min - 0.1, x_max + 0.1]
    ylim = [y_min - 0.1, y_max + 0.1]

    #Create gif for train
    for epoch in range(num_epochs):
        train_embeddings = all_embeddings['train'][epoch]
        train_labels = all_labels['train'][epoch]

        #Create a scatter plot with colored labels
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        

        for class_id in range(num_classes):
            class_indices = np.where(train_labels == class_id)[0]
            ax.scatter(train_embeddings[class_indices, 0], train_embeddings[class_indices, 1], label=f"Class {class_id}")

            # Add a unit circle to the plot
            circle = plt.Circle((0,0), 1, color='red', fill=False)
            ax.add_artist(circle)
        
        plt.title(f"Epoch [{epoch+1}/{num_epochs}] - Training Embeddings")
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        #Set the limits of the axes
        fig.gca().set_xlim([x_min-0.1, x_max+0.1])
        fig.gca().set_ylim([y_min-0.1, y_max+0.1])
        plt.legend()

        #Save the plot as an image
        plt.savefig('data/train_epoch_{}.png'.format(epoch))
        plt.close(fig)

        #Add the image to the list of frames
        train_frames.append(imageio.imread('data/train_epoch_{}.png'.format(epoch)))
    
    #Save the list of frames as a GIF
    imageio.mimsave('data/train_embeddings.gif', train_frames, duration=125)

    if visualize_val:

        #Get the min and max values for the val embeddings across all epochs
        x_min = np.min([np.min(all_embeddings['val'][epoch][:,0]) for epoch in range(num_epochs)])
        x_max = np.max([np.max(all_embeddings['val'][epoch][:,0]) for epoch in range(num_epochs)])
        y_min = np.min([np.min(all_embeddings['val'][epoch][:,1]) for epoch in range(num_epochs)])
        y_max = np.max([np.max(all_embeddings['val'][epoch][:,1]) for epoch in range(num_epochs)])

        # Set the limits of the axes
        xlim = [x_min - 0.1, x_max + 0.1]
        ylim = [y_min - 0.1, y_max + 0.1]

        #Create gif for val
        for epoch in range(num_epochs):
            val_embeddings = all_embeddings['val'][epoch]
            val_labels = all_labels['val'][epoch]

            #Create a scatter plot with colored labels
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            

            for class_id in range(num_classes):
                class_indices = np.where(val_labels == class_id)[0]
                ax.scatter(val_embeddings[class_indices, 0], val_embeddings[class_indices, 1], label=f"Class {class_id}")

                # Add a unit circle to the plot
                circle = plt.Circle((0,0), 1, color='red', fill=False)
                ax.add_artist(circle)
            
            plt.title(f"Epoch [{epoch+1}/{num_epochs}] - Validation Embeddings")
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            #Set the limits of the axes
            fig.gca().set_xlim([x_min-0.1, x_max+0.1])
            fig.gca().set_ylim([y_min-0.1, y_max+0.1])
            plt.legend()

            #Save the plot as an image
            plt.savefig('data/val_epoch_{}.png'.format(epoch))
            plt.close(fig)

            #Add the image to the list of frames
            val_frames.append(imageio.imread('data/val_epoch_{}.png'.format(epoch)))
        
        #Save the list of frames as a GIF
        imageio.mimsave('data/val_embeddings.gif', val_frames, duration=125)
