#!/usr/local/bin/python3

"""Train a bi-LSTM tagger using tensorFlow

Created on July 20, 2016
@author: Matthew Sevrens
@author: Oscar Pan
"""

############################################# USAGE ###############################################

# python3 -m meerkat.longtail.bilstm_tagger [config_file]
# python3 -m meerkat.longtail.bilstm_tagger meerkat/longtail/bilstm_config.json

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

from meerkat.various_tools import load_params, load_piped_dataframe

logging.basicConfig(level=logging.INFO)

def get_tags(trans):
	"""Convert df row to list of tags and tokens"""

	tokens = trans["Description"].split()
	tag = trans["Tagged_merchant_string"].split()
	tags = []

	for token in tokens:
		found = False
		for word in tag:
			if word in token and tag.index(word) == 0:
				tags.append("merchant")
				tag = tag[1:]
				found = True
				break
		if not found:
			tags.append("background")

	return (tokens, tags)

def validate_config(config):
	"""Validate input configuration"""

	config = load_params(config)

	return config

def load_data(config):
	"""Load labeled data"""

	df = load_piped_dataframe(config["dataset"])
	msk = np.random.rand(len(df)) < 0.90
	train = df[msk]
	test = df[~msk]

	return test, train

def load_embeddings_file(file_name, sep=" ",lower=False):
	"""
	load embeddings file
	"""
	emb={}
	for line in open(file_name):
		fields = line.split(sep)
		word = fields[0]
		if lower:
			word = word.lower()
		emb[word] = [float(x) for x in fields[1:]]

	print("loaded pre-trained embeddings (word->emb_vec) size: {} (lower: {})".format(len(emb.keys()), lower))
	return emb, len(emb[word])


def trans_to_tensor(tokens, tags, embedding):
	"""Convert a transaction to a tensor representation of documents
	and labels"""
	tensor = [np.asarray(embedding.get(word, [0]*64)) for word in tokens]
	return tensor

def evaluate_testset(config, graph, sess, model, test):
	"""Check error on test set"""

	test_accuracy = 0

	return test_accuracy

def build_graph(config):
	"""Build CNN"""

	graph = tf.Graph()
	# saver = tf.train.Saver()
	saver = {}

	return graph, saver

def train_model(config, graph, sess, saver):
	"""Train the model"""

	eras = config["eras"]
	dataset = config["dataset"]
	train, test = load_data(config)
	train_index = list(train.index)
	embedding, num_word = load_embeddings_file("./meerkat/longtail/embeddings/en.polyglot.txt")

	# Train the Model
	for step in range(eras):
		random.shuffle(train_index)
		for t_index in train_index:
			trans = train.loc[t_index]
			tokens, tags = get_tags(trans)
			trans, labels = trans_to_tensor(tokens, tags, embedding)

	final_model_path = ""

	return final_model_path

def run_session(config, graph, saver):
	"""Run Session"""

	model_path = config["model_path"]

	with tf.Session(graph=graph) as sess:

		tf.initialize_all_variables().run()
		train_model(config, graph, sess, saver)

def run_from_command_line():
	"""Run module from command line"""
	logging.basicConfig(level=logging.INFO)
	config = validate_config(sys.argv[1])
	graph, saver = build_graph(config)
	run_session(config, graph, saver)

if __name__ == "__main__":
	run_from_command_line()
