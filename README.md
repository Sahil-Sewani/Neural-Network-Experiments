# Neural Network Experiments with TensorFlow/Keras

## By: Sahil Sewani

This repository contains code that explores various aspects of neural network architectures using TensorFlow and Keras. The code is split into three main parts:

## Part 1: The Model

- Loads and preprocesses the CIFAR-10 dataset.
- Constructs a neural network model with three dense layers.
- Compiles the model and evaluates its performance.
- Visualizes training/validation accuracy and error over epochs.

## Part 2: Experiment 1 - Changing Numbers of Layers

- Investigates the impact of the number of layers on model performance.
- Creates and evaluates multiple neural network configurations with varying numbers of layers.

## Part 3: Experiment 2 - Changing Activation Functions

- Explores the effect of different activation functions on model performance.
- Constructs neural networks with different activation functions for each layer.
- Evaluates and compares training/test errors for each activation function.

## How to Use

1. **Environment Setup:** Ensure TensorFlow and necessary dependencies are installed.
2. **Run the Code:** 
   - Execute each part separately to conduct experiments and visualize results.
   - Comment/uncomment lines to switch between MNIST and CIFAR-10 datasets.
   
## Files Included

- `model.py`: Code for defining and training the neural network model.
- `experiment_layers.py`: Code to experiment with varying numbers of layers.
- `experiment_activations.py`: Code to explore different activation functions.

## Usage Notes

- The code is structured to allow for easy experimentation with different configurations.
- Each part contains detailed comments and explanations for better understanding.

## Acknowledgments

- This code is adapted from TensorFlow/Keras documentation and modified for experimentation purposes.

Feel free to explore the code, conduct further experiments, and modify it according to your requirements.
