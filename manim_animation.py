# 1. Import the required libraries
import numpy as np
import random
from manim import *


class PlotPoints(Scene):
    def __init__(self, num_points, num_iterations, **kwargs):
        super().__init__(**kwargs)

        self.num_points = num_points
        self.num_iterations = num_iterations

        # Generate random coordinates for each point for each iteration
        self.coordinates = np.zeros((self.num_points, self.num_iterations, 2))
        for i in range(self.num_points):
            for j in range(self.num_iterations):
                self.coordinates[i, j, 0] = random.uniform(-1, 1)
                self.coordinates[i, j, 1] = random.uniform(-1, 1)

        # Create a color map with random colors for each point
        self.colors = [random_color() for _ in range(self.num_points)]

    def construct(self):
        # Create dots at the initial positions for each point
        dots = []
        for i in range(self.num_points):
            dot = Dot(self.get_3d_coordinates(self.coordinates[i, 0]), color=self.colors[i])
            dots.append(dot)
            self.add(dot)

        # Store the transformations for each dot in a list
        transform_run_time = 2
        transforms = []
        for j in range(1, self.num_iterations):
            iteration_transforms = []
            for i in range(self.num_points):
                iteration_transforms.append(Transform(dots[i], Dot(self.get_3d_coordinates(self.coordinates[i, j]), color=self.colors[i]), run_time=transform_run_time))
            transforms.append(iteration_transforms)

            # Play the transformations simultaneously
            self.play(*iteration_transforms)
            self.wait(1)

    # A function to take as input 2d embedding and output a tuple of 3d coordinates (x, y, 0)
    def get_3d_coordinates(self, embedding):
        return (embedding[0], embedding[1], 0)


# Create an instance of the PlotPoints class
num_points = 6
num_iterations = 20
animation = PlotPoints(num_points, num_iterations)

# Play the animation
animation.render()

# # Set the output file name and path
# output_file = "data/animation.mp4"

# # Render and save the animation as a video
# animation.save(output_file)
