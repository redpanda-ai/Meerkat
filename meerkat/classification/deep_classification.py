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

import numpy as np
from pandas.io.parsers import read_csv
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
    input_shape=(None, 128),
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
    t = np.zeros((len(ALPHABET), l), dtype=np.int)
    for i, c in reversed(list(enumerate(s))):
        if c in ALPHABET:
            t[ALPHA_DICT[c]][len(s) - i + 1] = 1
    return t

# TEMPORARY EXAMPLE CODE
def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 9216),  # 96x96 input pixels per batch
    hidden_num_units=100,  # number of units in hidden layer
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=30,  # 30 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=400,  # we want to train this many epochs
    verbose=1,
)

X, y = load()

print(X.reshape(-1, 1, 96, 96).shape)

#net1.fit(X, y)

#TODO Character Quantization
#TODO Data Loading