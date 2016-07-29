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

from meerkat.classification.tools import reverse_map, get_tensor, get_op, get_variable
from meerkat.various_tools import load_params, load_piped_dataframe

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

def words_to_indices(data):
	"""convert tokens to int, assuming data is a df"""
	w2i = {}
	w2i["_UNK"] = 0
	for row_num, row in enumerate(data.values):
		tokens = row[0].lower().split()
		# there are too many unique tokens in description, better to shrink the size
		for token in tokens:
			if token not in w2i:
				w2i[token] = len(w2i)
	return w2i

def construct_embedding(config, w2i, loaded_embedding):
	"""construct an embedding that contains all words in loaded_embedding and w2i"""
	num_words = len(set(loaded_embedding.keys()).union(set(w2i.keys())))
	# initialize a num_words * we_dim embedding table
	temp = np.random.uniform(-1,1, (num_words, config["we_dim"]))
	for word in loaded_embedding.keys():
		if word not in w2i:
			w2i[word] = len(w2i)
		temp[w2i[word]] = loaded_embedding[word]
	return w2i, temp

def preprocess(config):
	"""Split data into training and test, return w2i for data in training, return training data's embedding matrix"""
	config["train"], config["test"] = load_data(config)
	embedding, emb_dim = load_embeddings_file(config["embeddings"], lower=True)
	# Assert that emb_dim is equal to we_dim
	assert(emb_dim == config["we_dim"])
	config["w2i"] = words_to_indices(config["train"])
	config["w2i"], config["wembedding"] = construct_embedding(config, config["w2i"], embedding)
	config["vocab_size"] = len(config["wembedding"])
	return config

def encode_tags(config, tags):
	"""one-hot encode labels"""
	tag2id = reverse_map(config["tag_map"])
	tags = np.array([int(tag2id[tag]) for tag in tags])
	encoded_tags = (np.arange(len(tag2id)) == tags[:, None]).astype(np.float32)
	return encoded_tags

def trans_to_tensor(config, sess, graph, tokens, tags, train=False):
	"""Convert a transaction to a tensor representation of documents
	and labels"""

	w2i = config["w2i"]
	c2i = config["c2i"]
	max_wl = config["max_word_length"]
	lst = get_tensor(graph, "last_state:0")
	rev_lst = get_tensor(graph, "rev_last_state:0")
	we_lookup = get_tensor(graph, "we_lookup:0")
	char_inputs, rev_char_inputs = [], []
	word_indices = [w2i[w] for w in tokens] if train else [w2i.get(w, w2i["_UNK"]) for w in tokens]

	# Lookup Character Indices
	for i, t in enumerate(tokens):

		t = ["<w>"] + list(t) + ["</w>"]
		rev_t = t[::-1]
		char_inputs.append([])
		rev_char_inputs.append([])

		for ii in range(max_wl):
			char_index = c2i[t[ii]] if ii < len(t) else 0
			char_inputs[i].append(char_index)
			rev_char_index = c2i[rev_t[ii]] if ii < len(rev_t) else 0
			rev_char_inputs[i].append(rev_char_index)

	# Collect Output
	feed_dict = {
		get_tensor(graph, "char_inputs:0") : np.array(char_inputs),
		get_tensor(graph, "rev_char_inputs:0") : np.array(rev_char_inputs),
		get_tensor(graph, "word_lengths:0") : [len(t) for t in tokens],
		get_tensor(graph, "word_inputs:0") : word_indices 
	}

	last_state, rev_last_state, embedded_words = sess.run([lst, rev_lst, we_lookup], feed_dict=feed_dict)
	char_embed = [last_state[i][len(t) - 1] for i, t in enumerate(tokens)]
	rev_char_embed = [rev_last_state[i][len(t) - 1] for i, t in enumerate(tokens)]

	# Merge Encodings
	char_features = [np.concatenate([c, rc], axis=0) for c, rc in zip(char_embed, reversed(rev_char_embed))]
	char_features = np.array(char_features)
	input_tensor = np.concatenate([embedded_words, char_features], axis=1)

	# Encode Tags
	encoded_tags = encode_tags(config, tags)

	print(tokens)
	print(input_tensor.shape) 

	return input_tensor, encoded_tags

def char_encoding(config, graph):
	"""Create graph nodes for character encoding"""

	c2i = config["c2i"]
	max_wl = config["max_word_length"]

	with graph.as_default():

		# Character Embedding
		word_lengths = tf.placeholder(tf.int32, [None], name="word_lengths")
		char_inputs = tf.placeholder(tf.int32, [None, max_wl], name="char_inputs")
		rev_char_inputs = tf.placeholder(tf.int32, [None, max_wl], name="rev_char_inputs")
		cembed_matrix = tf.Variable(tf.random_uniform([len(c2i.keys()), config["ce_dim"]], -1.0, 1.0), name="cembeds")
		cembeds = tf.nn.embedding_lookup(cembed_matrix, char_inputs, name="ce_lookup")
		rev_cembeds = tf.nn.embedding_lookup(cembed_matrix, rev_char_inputs, name="rev_ce_lookup")

		# TODO Non-essential: Replace extra with zeros instead of UNK embedding

		# Create LSTM for Character Encoding
		lstm = tf.nn.rnn_cell.BasicLSTMCell(config["ce_dim"])
		initial_state = lstm.zero_state(tf.size(word_lengths), tf.float32)

		# Encode Characters with LSTM
		options = {
			"dtype": tf.float32,
			"sequence_length": word_lengths,
			"initial_state": initial_state
		}

		output, state = tf.nn.dynamic_rnn(lstm, cembeds, **options)
		last_state = tf.identity(output, name="last_state")
		output, state =  tf.nn.dynamic_rnn(lstm, rev_cembeds, scope="rev", **options)
		rev_last_state = tf.identity(output, name="rev_last_state")

def build_graph(config):
	"""Build CNN"""

	graph = tf.Graph()
	c2i = config["c2i"]

	# Build Graph
	with graph.as_default():

		# Character Embedding
		char_encoding(config, graph)

		# Word Embedding
		word_inputs = tf.placeholder(tf.int32, [None], name="word_inputs")
		wembed_matrix = tf.Variable(tf.constant(0.0, shape=[config["vocab_size"], config["we_dim"]]), trainable=False, name="wembed_matrix")
		embedding_placeholder = tf.placeholder(tf.float32, [config["vocab_size"], config["we_dim"]], name="embedding_placeholder")
		assign_wembedding = tf.assign(wembed_matrix, embedding_placeholder, name="assign_wembedding")
		wembeds = tf.nn.embedding_lookup(wembed_matrix, word_inputs, name="we_lookup")
		
		# Combined Word Embedding and Character Embedding Input
		input_shape = (None, config["ce_dim"] * 2 + config["we_dim"])
		combined_embeddings = tf.placeholder(tf.float32, shape=input_shape, name="input")
		batched_input = tf.expand_dims(combined_embeddings, 0)

		# Create Model
		trans_len = tf.placeholder(tf.int64, None, name="trans_length")
		input_cell = tf.nn.rnn_cell.BasicLSTMCell(config["ce_dim"] * 2 + config["we_dim"])
		fw_lstm = tf.nn.rnn_cell.BasicLSTMCell(config["h_dim"])
		bw_lstm = tf.nn.rnn_cell.BasicLSTMCell(config["h_dim"])
		fw_network = tf.nn.rnn_cell.MultiRNNCell([input_cell] + [fw_lstm] * config["num_layers"])
		bw_network = tf.nn.rnn_cell.MultiRNNCell([input_cell] + [bw_lstm] * config["num_layers"])

		options = {
			"dtype": tf.float32,
			"sequence_length": tf.expand_dims(trans_len, 0)
		}

		(outputs_fw, outputs_bw), output_states = tf.nn.bidirectional_dynamic_rnn(fw_network, bw_network, batched_input, **options)
		tf.identity(outputs_fw, name="outputs_fw")
		tf.identity(outputs_bw, name="outputs_bw")

		saver = tf.train.Saver()

	return graph, saver

def train_model(config, graph, sess, saver):
	"""Train the model"""

	eras = config["eras"]
	dataset = config["dataset"]
	train, test = load_data(config)
	train_index = list(train.index)
	sess.run(get_op(graph, "assign_wembedding"), feed_dict={get_tensor(graph, "embedding_placeholder:0"): config["wembedding"]})

	# Train the Model
	for step in range(eras):
		random.shuffle(train_index)
		for t_index in train_index:
			trans = train.loc[t_index]
			tokens, tags = get_tags(trans)
			trans, labels = trans_to_tensor(config, sess, graph, tokens, tags)

			feed_dict = {
				get_tensor(graph, "trans_length:0"): trans.shape[0],
				get_tensor(graph, "input:0"): trans
			}

			outputs_fw, outputs_bw = sess.run([get_tensor(graph, "outputs_fw:0"), get_tensor(graph, "outputs_bw:0")], feed_dict=feed_dict)
			print(outputs_fw.shape)
			print(outputs_bw.shape)
			sys.exit()

	final_model_path = ""

	return final_model_path

def evaluate_testset(config, graph, sess, model, test):
	"""Check error on test set"""

	test_accuracy = 0

	return test_accuracy

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
	config = preprocess(config)
	graph, saver = build_graph(config)
	run_session(config, graph, saver)

if __name__ == "__main__":
	run_from_command_line()
