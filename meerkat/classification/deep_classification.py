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

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

ALPHABET = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}")
ALPHA_DICT = {}

# Create Alphabet Loopup
for i, a in enumerate(ALPHABET):
    ALPHA_DICT[a] = i

def character_encode(str, l):
    """Transform a transaction to a properly encoded representation"""
    s = str.lower()[0:l]
    t = np.zeros((len(ALPHABET), l), dtype=np.float32)
    for i, c in reversed(list(enumerate(s))):
        if c in ALPHABET:
            t[ALPHA_DICT[c]][len(s) - i - 1] = 1
    return t

def load_data(filename):
    """Loads transactions and encodes them into a proper format""" 
    
    l = 128
    df = pd.read_csv(filename, na_filter=False, encoding="utf-8", sep='|', error_bad_lines=False)
    labels = df["TRANSACTION_ORIGIN"].unique()
    label_map = dict(zip(labels, range(len(labels))))
    X = np.empty((len(df.index), len(ALPHABET), l))
    y = np.zeros(len(df.index), dtype=np.int32)

    for index, row in df.iterrows():
        X[index] = character_encode(row["DESCRIPTION_UNMASKED"], l)
        y[index] = label_map[row["TRANSACTION_ORIGIN"]]

    return X, y, len(labels)

X, y, output_units = load_data("data/input/bank_sample.txt")

deep_conv = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv1DLayer),
        ('pool1', layers.MaxPool1DLayer),
        ('conv2', layers.Conv1DLayer),
        ('pool2', layers.MaxPool1DLayer),
        ('conv3', layers.Conv1DLayer),
        ('conv4', layers.Conv1DLayer),
        ('conv5', layers.Conv1DLayer),
        ('conv6', layers.Conv1DLayer),
        ('pool3', layers.MaxPool1DLayer),
        ('hidden1', layers.DenseLayer),
        ('dropout1', layers.DropoutLayer),
        ('hidden2', layers.DenseLayer),
        ('dropout2', layers.DropoutLayer),
        ('output', layers.DenseLayer),
    ],
    input_shape=(None, len(ALPHABET), 128),
    conv1_num_filters=1, conv1_filter_length=7, pool1_ds=3,
    conv2_num_filters=1, conv2_filter_length=7, pool2_ds=3,
    conv3_num_filters=1, conv3_filter_length=3, 
    conv4_num_filters=1, conv4_filter_length=3,
    conv5_num_filters=1, conv5_filter_length=3, 
    conv6_num_filters=1, conv6_filter_length=3, pool3_ds=3,
    hidden1_num_units=512,
    hidden2_num_units=512,
    output_num_units=output_units,
    output_nonlinearity=lasagne.nonlinearities.softmax,
    update_learning_rate=0.01,
    dropout1_p=0.5,
    dropout2_p=0.5,
    update=nesterov_momentum,
    update_momentum=0.9,
    max_epochs=100,
    verbose=1
)

deep_conv.fit(X, y)
print(deep_conv.predict(X))