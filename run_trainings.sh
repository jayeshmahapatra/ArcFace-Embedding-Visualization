# Bash script to run the training of the models and visualize embeddings

# Run the training of the models
python train.py --model vgg8_arcface
python train.py --model vgg8_softmax

# Run the visualization script
python visualization.py --model vgg8_arcface
python visualization.py --model vgg8_softmax
