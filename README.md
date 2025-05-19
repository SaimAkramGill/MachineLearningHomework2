Machine Learning 1 - Homework 2: Neural Networks
This repository contains the implementation for the "Machine Learning 1" course homework assignment focused on Principal Component Analysis (PCA) and Neural Networks. The assignment involves image classification of MRI brain scans using both scikit-learn's MLPClassifier and a neural network implementation from scratch.
Dataset
The dataset consists of MRI brain scans classified into four categories:

No tumor
Glioma
Pituitary
Meningioma

The images are stored in NumPy format (.npy files):

brain_tumor_images.npy: Contains 64×64 pixel MRI scans
brain_tumor_targets.npy: Contains the class labels (0-3)

Project Structure
.
├── README.md
├── main.py                        # Main script to run all tasks
├── mlp_classifier_own.py          # Custom MLP classifier implementation
├── nn_classification_from_scratch.py  # Neural network training from scratch
├── nn_classification_sklearn.py   # Neural network using scikit-learn
├── requirements.txt               # Dependencies
├── autodiff/                      # Automatic differentiation framework
│   ├── __init__.py
│   ├── neural_net.py              # Neural network components
│   └── scalar.py                  # Scalar class for autodiff
└── data/                          # Dataset directory
    ├── brain_tumor_images.npy     # MRI scan images
    └── brain_tumor_targets.npy    # Target labels
Tasks
The assignment is divided into three main tasks:
Task 1: Neural Networks with scikit-learn

PCA for dimensionality reduction
Neural network training with different architectures
Regularization to prevent overfitting
Grid search for hyperparameter optimization
Evaluation metrics and confusion matrix

Task 2: Neural Networks From Scratch

Implementation of neurons, layers, and multi-layer perceptron
Forward and backward propagation
Custom loss functions
L2 regularization
Theoretical questions on backpropagation

Task 3: Binary Classification

Extension of the neural network for binary classification
Implementation of sigmoid activation and binary cross-entropy loss
Training on a binary subset of the dataset
Analysis of metrics for imbalanced datasets

Requirements
This project requires Python 3.11.5 with the following packages:

scikit-learn==1.6.1
numpy==2.2.3
matplotlib==3.10.1

You can install the dependencies using:
bashpip install -r requirements.txt
Usage
Run the full assignment:
bashpython main.py
To run specific tasks, modify the main() function in main.py to comment out unwanted tasks:
pythondef main():
    task_1()  # Neural Networks with scikit-learn
    task_2()    # Neural Networks From Scratch
    task_3()    # Binary Classification
Implementation Details
Neural Network Components

Neuron: Basic computational unit
FeedForwardLayer: Collection of neurons
MultiLayerPerceptron: Full network architecture

Activation Functions

ReLU for hidden layers
Softmax for multi-class classification
Sigmoid for binary classification

Loss Functions

Multi-class cross-entropy
Binary cross-entropy

Regularization

L2 regularization (weight decay)

Results
Multi-class Classification
The neural networks achieve approximately 85-90% accuracy on the test set for the multi-class problem, with the best architectures using regularization and moderate-sized hidden layers.
Binary Classification
For the binary classification task (distinguishing between "No tumor" and "Glioma"), the models achieve approximately 90% accuracy.
Author
[Muhammad Saim Akram] - Student at TU Graz, Summer Term 2025