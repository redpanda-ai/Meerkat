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
from tensorflow.models.rnn import rnn

from meerkat.various_tools import load_params, load_piped_dataframe
from meerkat.longtail.tools import get_tensor, get_op, get_variable

logging.basicConfig(level=logging.INFO)

def get_tags(trans):
	"""Convert df row to list of tags and tokens"""

	tokens = trans["Description"].lower().split()
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
	config["c2i"] = {a : i + 3 for i, a in enumerate(config["alphabet"])}
	config["c2i"]["_UNK"] = 0
	config["c2i"]["<w>"] = 1
	config["c2i"]["</w>"] = 2

	return config

def load_data(config):
	"""Load labeled data"""

	df = load_piped_dataframe(config["dataset"])
	msk = np.random.rand(len(df)) < 0.90
	train = df[msk]
	test = df[~msk]

	return test, train

def load_embeddings_file(file_name, sep=" ",lower=False):
	"""Load embeddings file"""

	emb = {}

	for line in open(file_name):
		fields = line.split(sep)
		word = fields[0]
		if lower:
			word = word.lower()
		emb[word] = [float(x) for x in fields[1:]]

	print("loaded pre-trained embeddings (word->emb_vec) size: {} (lower: {})".format(len(emb.keys()), lower))
	return emb, len(emb[word])

def trans_to_tensor(config, sess, graph, tokens, tags):
	"""Convert a transaction to a tensor representation of documents
	and labels"""

	c2i = config["c2i"]
	char_embed, rev_char_embed = [], []
	rev_lst = get_tensor(graph, "rev_last_state:0")
	lst = get_tensor(graph, "last_state:0")

	# Encode Characters
	for token in tokens:

		token = ["<w>"] + list(token) + ["</w>"]
		token_length = len(token)
		filler_zeros = np.random.rand(config["max_word_length"] - token_length, config["ce_dim"])

		feed_dict = {
			get_tensor(graph, "char_inputs:0") : [c2i[c] for c in token],
			get_tensor(graph, "word_length:0") : token_length,
			get_tensor(graph, "zeros:0") : filler_zeros
		}

		last_state, rev_last_state = sess.run([lst, rev_lst], feed_dict=feed_dict)
		char_embed.append(last_state[0][token_length-1])
		rev_char_embed.append(rev_last_state[0][token_length-1])

	# Encode Words

	# Merge Encodings
	features = [np.concatenate([c, rc], axis=0) for c, rc in zip(char_embed, reversed(rev_char_embed))]
	features = np.array(features)

	print(tokens)
	print(features.shape)
	sys.exit()
	tensor = []

	return tensor

def evaluate_testset(config, graph, sess, model, test):
	"""Check error on test set"""

	test_accuracy = 0

	return test_accuracy

def build_graph(config):
	"""Build CNN"""

	graph = tf.Graph()
	c2i = config["c2i"]

	# Create Graph
	with graph.as_default():

		# Character Embedding
		word_length = tf.placeholder(tf.int32, name="word_length")
		char_inputs = tf.placeholder(tf.int32, [None], name="char_inputs")
		filler_zeros = tf.placeholder(tf.float32, shape=[None, 100], name="zeros")
		cembed_matrix = tf.Variable(tf.random_uniform([len(c2i.keys()), config["ce_dim"]], -1.0, 1.0), name="cembeds")
		cembeds = tf.nn.embedding_lookup(cembed_matrix, char_inputs, name="ce_lookup")

		# Fill Cembeds with Padding Zeros #TODO: Simplify
		cembed_filler = tf.zeros_like(filler_zeros)
		cembed_filled = tf.concat(0, [cembeds, cembed_filler])
		ce_fixed_size = tf.expand_dims(cembed_filled, 0)
		rev_cembeds = tf.reverse(cembeds, [True, False])
		rev_cembed_filled = tf.concat(0, [rev_cembeds, cembed_filler])
		rev_ce_fixed_size = tf.expand_dims(rev_cembed_filled, 0)

		# Create LSTM for Character Encoding
		lstm = tf.nn.rnn_cell.BasicLSTMCell(config["ce_dim"])
		initial_state = lstm.zero_state(1, tf.float32)

		# Encode Characters with LSTM
		options = {
			"dtype": tf.float32,
			"sequence_length": tf.expand_dims(word_length, 0),
			"initial_state": initial_state
		}

		output, state = rnn.dynamic_rnn(lstm, ce_fixed_size, **options)
		last_state = tf.identity(output, name="last_state")
		output, state = rnn.dynamic_rnn(lstm, rev_ce_fixed_size, scope="rev", **options)
		rev_last_state = tf.identity(output, name="rev_last_state")
		
		# Combined Word Embedding and Character Embedding Input
		input_shape = (None, config["ce_dim"] * 2 + config["we_dim"])
		combined_embeddings = tf.placeholder(tf.float32, shape=input_shape, name="x")

		saver = tf.train.Saver()

	return graph, saver

def train_model(config, graph, sess, saver):
	"""Train the model"""

	eras = config["eras"]
	dataset = config["dataset"]
	train, test = load_data(config)
	train_index = list(train.index)
	# embedding, num_word = load_embeddings_file("./meerkat/longtail/embeddings/en.polyglot.txt")

	# Train the Model
	for step in range(eras):
		random.shuffle(train_index)
		for t_index in train_index:
			trans = train.loc[t_index]
			tokens, tags = get_tags(trans)
			trans, labels = trans_to_tensor(config, sess, graph, tokens, tags)

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
