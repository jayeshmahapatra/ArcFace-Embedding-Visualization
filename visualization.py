import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from PIL import Image
import imageio
import argparse


def visualize_embeddings(all_embeddings, all_labels, visualize_val=False, args=None):
    
    # First plot train embeddings
    plot_embeddings(all_embeddings['train'], all_labels['train'], title_poststr="Training_Embeddings", args=args)

    if visualize_val:
        # Then plot val embeddings
        plot_embeddings(all_embeddings['val'], all_labels['val'], title_poststr="Validation_Embeddings", args=args)


# Function to plot the embeddings using matplotlib
def plot_embeddings(embeddings, labels, title_poststr = "Training_Embeddings", args=None):
    
    #Get the number of epochs
    num_epochs = len(embeddings)

    #Num classes
    num_classes = len(np.unique(labels[0]))

    # Create a GIF of the embeddings

    #Create list to hold frames for train and val
    frames = []


    #Create gif for train
    for epoch in range(num_epochs):
        epoch_embeddings = embeddings[epoch]
        epoch_labels = labels[epoch]

        #Create a scatter plot with colored labels
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection='3d')

        for class_id in range(num_classes):
            class_indices = np.where(epoch_labels == class_id)[0]

            # Get current embedding as normalized embedding
            curr_embeddings = epoch_embeddings[class_indices]
            curr_embeddings = curr_embeddings / np.linalg.norm(curr_embeddings, axis=1, keepdims=True)

            # Plot the embeddings
            ax.scatter(curr_embeddings[:, 0], curr_embeddings[:, 1], curr_embeddings[:, 2], label=f"Class {class_id}", alpha=0.5)

        ax.set(xlabel='Dimension 1', ylabel='Dimension 2', zlabel='Dimension 3')
        plt.title(f"Epoch [{epoch+1}/{num_epochs}] - {title_poststr}")

        plt.legend()

        #Save the plot as an image
        plt.savefig(f"data/{args.model}_{title_poststr}_{epoch}.png")
        plt.close(fig)

        #Add the image to the list of frames
        frames.append(imageio.imread(f"data/{args.model}_{title_poststr}_{epoch}.png"))
    
    #Save the list of frames as a GIF
    imageio.mimsave(f"data/{args.model}_{title_poststr}.gif", frames, duration=125)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a model on MNIST dataset')

    parser.add_argument('--model', type=str, default='vgg8_arcface', help='Model to train')

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()

    #Load the embeddings and labels
    all_embeddings = np.load(f"data/all_{args.model}_embeddings.npy", allow_pickle=True).item()
    all_labels = np.load(f"data/all_{args.model}_labels.npy", allow_pickle=True).item()

    #Visualize the embeddings
    visualize_embeddings(all_embeddings, all_labels, visualize_val=True, args=args)
