#!/usr/bin/python

#########################################
# module: cs5600_6600_f24_hw02.py
# description: some starter code
# bugs to vladimir kulyukin in canvas.
#########################################

import numpy as np
import pickle
from cs5600_6600_f24_hw02_data import *

# sigmoid function
def sigmoid(x, deriv=False):
    if(deriv):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))
    
# save() function to save the trained network to a file
def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(obj, fp)

# restore() function to restore the file
def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def build_nn_wmats(mat_dims):
    matrix = []
    if len(mat_dims) == 2:
        matrix.append(build_weight_matrix(mat_dims[0], mat_dims[1]))
    else:
        # Create weight matrices for each pair of consecutive layers
        for i in range(len(mat_dims) - 1):  # Loop to create weight matrices
            matrix.append(build_weight_matrix(mat_dims[i], mat_dims[i + 1]))
    
    return matrix
def build_weight_matrix(rows, cols):
    return np.random.randn(rows, cols)

def build_231_nn():
    return build_nn_wmats((2, 3, 1))

def build_2331_nn():
    return build_nn_wmats((2, 3, 3, 1))

def build_221_nn():
    return build_nn_wmats((2, 2, 1))

def build_838_nn():
    return build_nn_wmats((8, 3, 8))

def build_949_nn():
    return build_nn_wmats((9, 4, 9))

def build_4221_nn():
    return build_nn_wmats((4, 2, 2, 1))

def build_421_nn():
    return build_nn_wmats((4, 2, 1))

def build_121_nn():
    return build_nn_wmats((1, 2, 1))

def build_1221_nn():
    return build_nn_wmats((1, 2, 2, 1))

## Training 3-layer neural net.
## X is the matrix of inputs
## y is the matrix of ground truths.
## build is a nn builder function.
def train_3_layer_nn(numIters, X, y, build):
    wmats = build()
    validate_input_output_match(X,y)
    for i in range(0, numIters-1):
        a1 = X
        z2 = np.dot(a1, wmats[0])
        a2 = sigmoid(z2)
        z3 = np.dot(a2, wmats[1])
        a3 = sigmoid(z3)
        error = y - a3
        d3 = error * sigmoid(a3, True)

        error_hidden = d3.dot(wmats[1].T)
        d2 = error_hidden * sigmoid(a2, True)

        wmats[1] += a2.T.dot(d3) * 0.3
        wmats[0] += a1.T.dot(d2) * 0.3

    return wmats

def train_4_layer_nn(numIters, X, y, build):
    wmats = build()
    validate_input_output_match(X,y)
    for iter in range(numIters):
        a1 = X
        z2 = np.dot(a1, wmats[0])
        a2 = sigmoid(z2)  # First hidden layer activations
        
        z3 = np.dot(a2, wmats[1])
        a3 = sigmoid(z3)  # Second hidden layer activations
        
        z4 = np.dot(a3, wmats[2])
        a4 = sigmoid(z4)  # Output layer activations

        # Backpropagation
        error_output = y - a4
        delta4 = error_output * sigmoid(a4, deriv=True)  # Delta for output layer
        
        error_hidden2 = np.dot(delta4, wmats[2].T)
        delta3 = error_hidden2 * sigmoid(a3, deriv=True)  # Delta for second hidden layer
        
        error_hidden1 = np.dot(delta3, wmats[1].T)
        delta2 = error_hidden1 * sigmoid(a2, deriv=True)  # Delta for first hidden layer
        
        # Update weights
        wmats[2] += np.dot(a3.T, delta4)
        wmats[1] += np.dot(a2.T, delta3)
        wmats[0] += np.dot(a1.T, delta2)

    return wmats

def fit_3_layer_nn(x, wmats, thresh=0.4, thresh_flag=False):
    a1 = x
    
    z2 = np.dot(a1, wmats[0])
    a2 = sigmoid(z2)
    
    z3 = np.dot(a2, wmats[1])
    a3 = sigmoid(z3)

    if thresh_flag:
        a3 = (a3 > thresh).astype(int)
    
    return a3
def fit_4_layer_nn(x, wmats, thresh=0.4, thresh_flag=False):
    a1 = x
    
    z2 = np.dot(a1, wmats[0])
    a2 = sigmoid(z2)
    
    z3 = np.dot(a2, wmats[1])
    a3 = sigmoid(z3)

    z4 = np.dot(a3, wmats[2])
    a4 = sigmoid(z4)

    if thresh_flag:
        a4 = (a4 > thresh).astype(int)
    
    return a4

def validate_input_output_match(X, y):
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Number of samples in X ({X.shape[0]}) and y ({y.shape[0]}) do not match.")
    
    if len(X.shape) != 2:
        raise ValueError(f"X should be a 2D array, but got shape {X.shape}.")
    if len(y.shape) not in [1, 2]:
        raise ValueError(f"y should be a 1D or 2D array, but got shape {y.shape}.")

    print("Input X and ground truth y match for training.")
    return True


