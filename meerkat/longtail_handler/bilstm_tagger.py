#!/usr/local/bin/python3

"""Train a bi-LSTM tagger using tensorFlow

Created on July 20, 2016
@author: Matthew Sevrens
@author: Oscar Pan
"""

############################################# USAGE ###############################################

# meerkat.longtail_handler.bilstm_tagger [config_file]
# meerkat.longtail_handler.bilstm_tagger bilstm_config.json

# For addtional details on implementation see:
#
# Multilingual Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Models and Auxiliary Loss
# http://arxiv.org/pdf/1604.05529v2.pdf
# https://github.com/bplank/bilstm-aux
# https://github.com/KnHuq/Tensorflow-tutorial/blob/master/BiDirectional%20LSTM/bi_directional_lstm.ipynb
#
# Deep Learning for Character-based Information Extraction
# http://www.cs.cmu.edu/~qyj/zhSenna/2014_ecir2014_full.pdf
#
# Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Recurrent Neural Network
# http://arxiv.org/pdf/1510.06168v1.pdf
#
# Finding Function in Form: Compositional Character Models for Open Vocabulary Word Representation
# http://arxiv.org/pdf/1508.02096v2.pdf

###################################################################################################

import logging
import math
import os
import pprint
import random
import sys

import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

def validate_config(config):
	"""Validate input configuration"""

	config = load_params(config)

	return config

def load_data(config):
	"""Load labeled data"""

	data = ""

	return data

def trans_to_tensor(trans):
	"""Convert a transaction to a tensor representation"""

	parsed_transaction = []

	return parsed_transaction

def evaluate_testset(config, graph, sess, model, test):
	"""Check error on test set"""

	test_accuracy = 0

	return test_accuracy

def build_graph(config):
	"""Build CNN"""

	graph = tf.Graph()
	saver = tf.train.Saver()

	return graph, saver

def train_model(config, graph, sess):
	"""Train the model"""

	final_model_path = ""

	return final_model_path

def run_session(config, graph, saver):
	"""Run Session"""

	with tf.Session(graph=graph) as sess:

		model_path = config["model_path"]

def run_from_command_line():
	"""Run module from command line"""
	logging.basicConfig(level=logging.INFO)
	config = validate_config(sys.argv[1])
	graph, saver = build_graph(config)
	run_session(config, graph, saver)

if __name__ == "__main__":
	run_from_command_line()
