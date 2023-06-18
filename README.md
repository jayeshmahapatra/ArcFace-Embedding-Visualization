
# 3D ArcFace Embeddings Visualization

This repository contains code for code for creating 3D ArcFace embeddings (projected to a unit sphere) and visualizing their separation during training. The goal is to demonstrate how different classes get separated from each other through the course of training using the ArcFace loss.

<p align="center">
   <figure>
   <img src="data/Training_Embeddings.gif" width="500" height="500"/>
   <figcaption>Fig.1 - A GIF demonstrating how 3d embeddings evolve during training.</figcaption>
   </figure>
</p>

This is a companion repo to my blog about using ArcFace Loss.<br />
We train a VGG8 Network with ArcFace Loss to generate these embeddings.

## Usage

To use this code to produce and visualize embeddings:

#### 1. Clone the repository.
```
git clone https://github.com/jayeshmahapatra/ArcFace-Embedding-Visualization.git
```

#### 2. Install the required dependencies.
```
pip install -r requirements.txt
```

#### 3. Run the training script to train a VGG8 model on the `MNIST` dataset. This will download the `MNIST` dataset and train the model on that.

```
python train.py
```

This will also save the training and validation embeddings generated during the training. The embeddings are stored at:

- data/all_embeddings.npy
- data/all_labels.npy


#### 4. Run the visualization script to generate gifs visualizing how the embeddings evolved through the course the model training.

```
python visualization.py
```

This script will generate plots for how embeddings looked at each epoch, and then combine these plots to generate gifs. These gifs are stored at:


- data/Training_Embeddings.gif
- data/Validation_Embeddings.gif

## Contributing

Contributions to this repository are welcome! If you find any issues or have suggestions for improvements, please create a new issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

