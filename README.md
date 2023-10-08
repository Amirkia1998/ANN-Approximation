# Neural Network Models for Data Approximation

This repository contains MATLAB code that demonstrates the use of various neural network models for data approximation. In particular, we explore the use of Radial Basis Function (RBF) networks, Feedforward Backpropagation networks, and Cascade Forward Backpropagation networks to approximate a given dataset.

## Problem Presentation
The dataset used in this example is represented by two-dimensional points. Our goal is to approximate this dataset using different neural network architectures and examine their performance.


## Data Sorting
Before we proceed, we sort the dataset based on the x-coordinate to ensure consistency in our experiments.


## Choosing the Training Dataset
We randomly select a subset of points from the sorted dataset to create our training data. The number of points to choose is controlled by the `hm` variable.

## Preparing Training and Testing Data
We split the dataset into training and testing data, where the training data consists of the chosen points, and the testing data includes all points from the sorted dataset. This division allows us to train the neural networks on a subset of the data and evaluate their performance on the entire dataset.

## Radial Basis Function Networks (RBF)
We explore the use of RBF networks with varying spread values (1.0, 2.0, 0.5, 5.0, and 10.0) to approximate the dataset.

### RBF Network Training
We train the RBF networks to minimize the differences between the predicted and target output.

### Network Structure
For each RBF network, we inspect the following properties:
- Number of inputs
- Number of layers
- Number of outputs
- Number of neurons in Layer 1
- Number of neurons in Layer 2 (if applicable)
- Weights and biases for each layer

### Results
We compare the performance of RBF networks with different spread values and observe their output for the entire dataset.

## Feedforward Backpropagation Networks (FFBP)
We use Feedforward Backpropagation networks with different architectures (12-tansig-1-purelin and 8-tansig-4-tansig-1-purelin) to approximate the dataset.

### FFBP Network Training
We train the FFBP networks to minimize the output error.

### Network Structure
For each FFBP network, we inspect the following properties:
- Number of inputs
- Number of layers
- Number of outputs
- Number of neurons in Layer 1
- Number of neurons in Layer 2 (if applicable)
- Weights and biases for each layer

### Results
We compare the performance of FFBP networks with different architectures and visualize their output for the entire dataset.

## Cascade Forward Backpropagation Networks (CFBP)
We use Cascade Forward Backpropagation networks with a 12-tansig-1-purelin architecture to approximate the dataset.

### CFBP Network Training
We train the CFBP network to minimize the output error.

### Network Structure
For the CFBP network, we inspect the following properties:
- Number of inputs
- Number of layers
- Number of outputs
- Number of neurons in Layer 1
- Number of neurons in Layer 2 (if applicable)
- Weights and biases for each layer

### Results
We compare the performance of the CFBP network with other models and visualize its output for the entire dataset.

## Conclusion
This code demonstrates the use of different neural network architectures for data approximation and provides insights into their performance. You can explore the code and experiment with various network configurations to see how they affect the accuracy of data approximation.

Feel free to adapt this code for your own datasets and experiments, and don't hesitate to reach out with any questions or feedback. Happy coding!

**Note:** In neural networks, transfer functions are used to introduce nonlinearity into the model. The transfer functions used in this code are the tanh (hyperbolic tangent) and purelin (linear) functions. Tanh maps the input to a value between -1 and 1, while purelin is the identity function. Tanh is typically used in hidden layers, while purelin is used in the output layer when a linear combination of inputs is desired.

## Outputs

![img](https://github.com/Amirkia1998/ANN-Approximation/blob/main/ANN-Approximation/Capture.PNG)
