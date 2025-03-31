# Multilayer-Perceptron
Machine learning model using a Multilayer Perceptron (MLP) architecture for handwritten digit classification on the MNIST datase

Model classifies handwritten digits using  MNIST dataset which contains handwritten digits with a training set of 60,000 examples, and a test set of 10,000 examples.

Trained using a 3 layered network

$x → h 1 → h 2 → p ( y | h 2 )$

where x is the input, h1 and h2 are the hidden layers, and p(y|h2) is the output layer.

The hidden layers h 1 and h 2 have dimensions 500 . The network is trained for 250 epochs and tests the classification error. 

Uses the sigmoid activation function for the hidden layers and the softmax activation function for the output layer. 

Uses the SGD optimizer with a learning rate of 0.01 and a batch size of 1000.

Uses a validation split to 0.4 as well as the cross entropy loss function.

The program then plots the cross entropy loss on the batches and the classification error on the validation data.

The model then repeast the experiment with the following changes:
        - Uses the ReLU activation function for the hidden layers and the softmax activation function for the output layer.
        - Uses the L 2 regularization with a lambda of 0.01 .
        - Uses dropout with a probability of 0.5 for each of the two hidden layers.
        - User early stopping based on monitoring the validation loss. Set the patience to 1 epochs. 

The model shows that Adding L2 regularization to the Sigmoid model reduced overfitting, but it also slightly increased validation error due to underfitting

For the ReLU model, L2 regularization improved generalization, which resulted in a lower validation error compared to the baseline ReLU model

Dropout in the sigmoid model reduced overfitting but increased validation error.

Early stopping effectively prevented overfitting in both Sigmoid and Relu models by stopping training when validation loss stopped improving.

Overall regularization techniques improved generalization. Early stopping proved beneficial for both activation functions and it helped achieve a higher model performance overall.
