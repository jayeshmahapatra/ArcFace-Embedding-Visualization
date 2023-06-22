
# 3D ArcFace Embeddings Visualization

This repository contains code for visualizing how embeddings evolve when trained using ArcFace and Softmax Loss.

<div align="center">
   <h3>Embedding Comparison</h3>
   <div style="display: flex; justify-content: center;">
      <div style="margin-right: 10px;">
         <img src="data/vgg8_arcface_Training_Embeddings.gif" width="400" height="400" />
         <figcaption>ArcFace 3D embeddings during training.</figcaption>
      </div>
      <div style="margin-left: 10px;">
         <img src="data/vgg8_softmax_Training_Embeddings.gif" width="400" height="400" />
         <figcaption>Softmax 3D embeddings during training.</figcaption>
      </div>
   </div>
</div>

<br />
<br />
This is a companion repo to my blog about using ArcFace Loss.<br />
We train a VGG8 Network with ArcFace Loss to generate these embeddings.
<br />

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

#### 3. Run the training scripts to train a VGG8 model on the `MNIST` dataset. This will download the `MNIST` dataset and train the model on that. You can use the `--model` parameter to specify whether to train arcface or softmax model.

```
python train.py --model vgg8_arcface
python train.py --model vgg8_softmax
```

This will also save the training and validation embeddings generated during the training. The embeddings are stored at:

- data/all_{model_name}_embeddings.npy
- data/all_{model_name}_labels.npy

Where *model_name* can be `vgg8_arcface` or `vgg8_softmax`.


#### 4. Run the visualization script to generate gifs visualizing how the embeddings evolved through the course the model training. You can use the `--model` parameter to specify whether to generate visualizations of the arcface or softmax embeddings.

```
python visualization.py --model vgg8_arcface
python visualization.py --model vgg8_softmax
```

This script will generate plots for how embeddings looked at each epoch, and then combine these plots to generate gifs. These gifs are stored at:


- data/{model_name}_Training_Embeddings.gif
- data/{model_name}_Validation_Embeddings.gif

Where *model_name* can be `vgg8_arcface` or `vgg8_softmax`.

## Replicate results in one line
If you want to just reproduce the gifs above, I have packaged all commands into a single bash script.
```
bash run_all.sh
```
This should run the model trainings and then run the visualization scripts to generate the gifs.

## Contributing

Contributions to this repository are welcome! If you find any issues or have suggestions for improvements, please create a new issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

