#!/usr/local/bin/python3.3

"""This module trains a 1D deep temporal convnet for usage in transaction
related classification tasks. Included in a character quantization method
as opposed to bag of words or other high dimensional techniques for
featurization. 

Created on Mar 31, 2015
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# sudo python3 -m meerkat.classification.deep_classification

#####################################################

import os
import sys

import pandas as pd
import numpy as np
from sklearn.utils import shuffle

from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

FTRAIN = 'data/input/training.csv'
FTEST = 'data/input/test.csv'
ALPHABET = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}")
ALPHA_DICT = {}

# Create Alphabet Loopup
for i, a in enumerate(ALPHABET):
    ALPHA_DICT[a] = i

deep_conv = NeuralNet(
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
    input_shape=(128, len(ALPHABET), 128),
    output_num_units=500,
    update_learning_rate=0.01,
    update=nesterov_momentum,
    update_momentum=0.9,
    max_epochs=500,
    verbose=1
)

def character_encode(str, l):
    """Transform a transaction to a properly encoded representation"""
    s = str.lower()[0:l]
    t = np.zeros((len(ALPHABET), l), dtype=np.float)
    for i, c in reversed(list(enumerate(s))):
        if c in ALPHABET:
            t[ALPHA_DICT[c]][len(s) - i - 1] = 1
    return t

def load_data(filename):
    """Loads transactions and encodes them into a proper format""" 
    
    l = 128
    df = pd.read_csv(filename, na_filter=False, encoding="utf-8", sep='|', error_bad_lines=False)
    X = np.empty((len(df.index), len(ALPHABET), l))
    y = np.zeros((len(df.index)), dtype=np.int)

    for index, row in df.iterrows():
        X[index] = character_encode(row["DESCRIPTION_UNMASKED"], l)

    print(X.shape)

    return X, y

X, y = load_data("data/misc/transaction_type_GT_Bank.txt")
#deep_conv.fit(X, y)

#TODO Load labels into numpy array