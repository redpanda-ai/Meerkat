#!/usr/local/bin/python3.3

"""This module trains a 1D deep temporal convnet for usage in transaction
related classification tasks. Included in a character quantization method
as opposed to bag of words or other high dimensional techniques for
featurization. 

Created on Mar 31, 2015
@author: Matthew Sevrens
"""

from lasagne import layers
from nolearn.lasagne import NeuralNet

deep = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv1DLayer),
        ('pool1', layers.MaxPool1DLayer),
        ('conv2', layers.Conv1DLayer),
        ('pool2', layers.MaxPool1DLayer),
        ('conv3', layers.Conv1DLayer),
        ('pool3', layers.MaxPool1DLayer),
        ('conv3', layers.Conv1DLayer),
        ('pool4', layers.MaxPool1DLayer),
        ('conv5', layers.Conv1DLayer),
        ('pool5', layers.MaxPool1DLayer),
        ('conv6', layers.Conv1DLayer),
        ('pool6', layers.MaxPool1DLayer),
        ('hidden7', layers.DenseLayer),
        ('hidden8', layers.DenseLayer),
        ('hidden9', layers.DenseLayer),
        ('output', layers.DenseLayer),
    ],
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=500,
    verbose=1
)