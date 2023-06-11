# 1. Import the required libraries
import numpy as np
import random
from manim import *


class PlotPoints(Scene):
    def __init__(self, all_embeddings, all_labels, num_samples = 10, **kwargs):
        super().__init__(**kwargs)

        # Store the embeddings and labels
        self.all_embeddings = all_embeddings
        self.all_labels = all_labels

        self.num_epochs = len(all_embeddings['train'])
        self.num_classes = len(np.unique(all_labels['train'][0]))
        self.num_samples = num_samples

        # Calculate the maximum absolute coordinate x, y value among all the embeddings
        self.max_x = np.max(np.abs(all_embeddings['train'][0][:,0]))
        self.max_y = np.max(np.abs(all_embeddings['train'][0][:,1]))

        # Create a color map with random colors for each point, making sure colors don't repeat between points
        self.colors = []
        color_set = set()
        while len(self.colors) < self.num_classes:
            color = random_color()
            if color not in color_set:
                self.colors.append(color)
                color_set.add(color)

        # Create axes centered at the origin
        self.axes = Axes(
            x_range=[-self.max_x - 0.1, self.max_x + 0.1, 1],
            y_range=[-self.max_y - 0.1, self.max_y + 0.1, 1],
            x_length=8,
            #y_length=8,
            tips=False,
            # add numbers to the axes
            x_axis_config={"numbers_to_include": np.arange(-self.max_x, self.max_x + 1)},
            y_axis_config={"numbers_to_include": np.arange(-self.max_y, self.max_y + 1)},

        )

        # Print all the class variables
        print(f"Number of epochs: {self.num_epochs}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Number of samples: {self.num_samples}")
        print(f"Maximum x value: {self.max_x}")
        print(f"Maximum y value: {self.max_y}")
        print(f"Colors: {self.colors}")

        

    def construct(self):

        # Add the axes to the scene
        self.add(self.axes)



        # Create dots at the first epoch positions for embeddings of each class
        dots = []
        for class_id in range(self.num_classes):
            #Get indices of embeddings belonging to the current class
            indices = np.where(self.all_labels['train'][0] == class_id)[0]
            for i in range(self.num_samples):
                # Create a dot at the 3d coordinates of the embedding
                dot = Dot(self.get_3d_coordinates(self.all_embeddings['train'][0][indices[i]]), color=self.colors[class_id])
                dots.append(dot)
                self.add(dot)
        

        # Create a legend specifying class to color mapping
        legend_entries = [
            Text(f"Class {class_id}", color=self.colors[class_id])
            for class_id in range(self.num_classes)
        ]
        legend = Group(*legend_entries)
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        legend.scale(0.8)
        legend.shift(RIGHT * 5 + UP * 3)
        self.add(legend)

        # Store the transformations for each dot in a list
        transform_run_time = 0.5
        transforms = []

        # iterate over number of epochs
        for j in range(1, self.num_epochs):
            iteration_transforms = []
            # iterate over num classes
            for class_id in range(self.num_classes):
                #Get the indices of embeddings belonging to the current class
                indices = np.where(self.all_labels['train'][j] == class_id)[0]

                # iterate over num samples
                for i in range(self.num_samples):
                    iteration_transforms.append(Transform(dots[self.num_samples * class_id + i], Dot(self.get_3d_coordinates(self.all_embeddings['train'][j][indices[i]]), color=self.colors[class_id]), run_time=transform_run_time))
                    #iteration_transforms.append(Transform(dots[self.num_samples * class_id + i], Dot(self.get_3d_coordinates(all_embeddings['train'][j][i]), color=self.colors[class_id]), run_time=transform_run_time))

            transforms.append(iteration_transforms)

            # Play the transformations simultaneously
            self.play(*iteration_transforms)
            self.wait(0.25)

    # A function to take as input 2d embedding and output a tuple of 3d coordinates (x, y, 0)
    def get_3d_coordinates(self, embedding):
        x_mapped = self.axes.c2p(embedding[0], 0)[0]
        y_mapped = self.axes.c2p(0, embedding[1])[1]
        return (x_mapped, y_mapped, 0)
        #return (embedding[0], embedding[1], 0)



if __name__ == "__main__":

    # Create an instance of the PlotPoints class

    # Create a dictionary of embeddings and labels for each epoch
    all_embeddings = {'train': []}
    all_labels = {'train': []}

    # Generate random embeddings and labels for each epoch. Labels can belong to 6 classes
    for i in range(3):
        all_embeddings['train'].append(np.random.rand(9, 2))
        # generate a repeating array containing 3 class labels 10 times each
        repeated_labels = np.tile(np.arange(3), 3)
        # suffle the labels and append to the list
        all_labels['train'].append(np.random.permutation(repeated_labels))

    # print embeddings and labels side by side
    for i in range(3):
        print(all_embeddings['train'][i], all_labels['train'][i])


    animation = PlotPoints(all_embeddings, all_labels, num_samples=3)


    # Play the animation
    animation.render()

    # # Set the output file name and path
    # output_file = "data/animation.mp4"

    # # Render and save the animation as a video
    # animation.save(output_file)
