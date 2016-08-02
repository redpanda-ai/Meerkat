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

def last_relevant(output, length, name):
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    output_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (tf.to_int32(length) - 1)
    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, index, name=name)
    return relevant

def get_tags(trans):
	"""Convert df row to list of tags and tokens"""

	tokens = trans[0].lower().split()
	tag = trans[1].lower().split()
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

	# TODO: Fix this hack
	if len(tokens) == 1:
		return(tokens * 2, tags * 2)

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

	return train, test

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
	char_inputs, rev_char_inputs = [], []
	word_indices = [w2i[w] for w in tokens] if train else [w2i.get(w, w2i["_UNK"]) for w in tokens]

	# Encode Tags
	encoded_tags = encode_tags(config, tags)

	# Lookup Character Indices
	for i, t in enumerate(tokens):

		t = ["<w>"] + list(t) + ["</w>"]
		char_inputs.append([])

		for ii in range(max_wl):
			char_index = c2i[t[ii]] if ii < len(t) else 0
			char_inputs[i].append(char_index)

	char_inputs = np.array(char_inputs)
	word_lengths = [len(t) for t in tokens]

	return char_inputs, word_lengths, word_indices, encoded_tags

def char_encoding(config, graph):
	"""Create graph nodes for character encoding"""

	c2i = config["c2i"]
	max_wl = config["max_word_length"]

	with graph.as_default():

		# Character Embedding
		word_lengths = tf.placeholder(tf.int64, [None], name="word_lengths")
		char_inputs = tf.placeholder(tf.int32, [None, max_wl], name="char_inputs")
		cembed_matrix = tf.Variable(tf.random_uniform([len(c2i.keys()), config["ce_dim"]], -0.25, 0.25), name="cembeds")
		cembeds = tf.nn.embedding_lookup(cembed_matrix, char_inputs, name="ce_lookup")

		# Create LSTM for Character Encoding
		fw_lstm = tf.nn.rnn_cell.BasicLSTMCell(config["ce_dim"], state_is_tuple=True)
		bw_lstm = tf.nn.rnn_cell.BasicLSTMCell(config["ce_dim"], state_is_tuple=True)

		# Encode Characters with LSTM
		options = {
			"dtype": tf.float32,
			"sequence_length": word_lengths
		}

		(output_fw, output_bw), output_states = tf.nn.bidirectional_dynamic_rnn(fw_lstm, bw_lstm, cembeds, **options)
		last_state = tf.identity(output_fw, name="last_state")
		rev_last_state = tf.identity(output_bw, name="rev_last_state")

		return last_state, rev_last_state, word_lengths

def build_graph(config):
	"""Build CNN"""

	graph = tf.Graph()
	c2i = config["c2i"]

	# Build Graph
	with graph.as_default():

		# Character Embedding
		last_state, rev_last_state, word_lengths = char_encoding(config, graph)

		# Word Embedding
		word_inputs = tf.placeholder(tf.int32, [None], name="word_inputs")
		wembed_matrix = tf.Variable(tf.constant(0.0, shape=[config["vocab_size"], config["we_dim"]]), trainable=True, name="wembed_matrix")
		embedding_placeholder = tf.placeholder(tf.float32, [config["vocab_size"], config["we_dim"]], name="embedding_placeholder")
		assign_wembedding = tf.assign(wembed_matrix, embedding_placeholder, name="assign_wembedding")
		wembeds = tf.nn.embedding_lookup(wembed_matrix, word_inputs, name="we_lookup")
		trans_len = tf.placeholder(tf.int64, None, name="trans_length")

		# Combine Embeddings
		char_embeds = last_relevant(last_state, word_lengths, "char_embeds")
		rev_char_embeds = last_relevant(rev_last_state, word_lengths, "rev_char_embeds")
		combined_embeddings = tf.concat(1, [wembeds, char_embeds, tf.reverse(rev_char_embeds, [True, False])], name="combined_embeddings")

		# Cells and Weights
		input_cell = tf.nn.rnn_cell.BasicLSTMCell(config["ce_dim"] * 2 + config["we_dim"], state_is_tuple=True)
		fw_lstm = tf.nn.rnn_cell.BasicLSTMCell(config["h_dim"], state_is_tuple=True)
		bw_lstm = tf.nn.rnn_cell.BasicLSTMCell(config["h_dim"], state_is_tuple=True)

		fw_network = tf.nn.rnn_cell.MultiRNNCell([input_cell] + ([fw_lstm] * (config["num_layers"] - 1)), state_is_tuple=True)
		bw_network = tf.nn.rnn_cell.MultiRNNCell([input_cell] + ([bw_lstm] * (config["num_layers"] - 1)), state_is_tuple=True)
		weight = tf.Variable(tf.truncated_normal([config["h_dim"] * 2, len(config["tag_map"])], stddev=0.1), name="weight")
		bias = tf.Variable(tf.constant(0.1, shape=[len(config["tag_map"])]))

		def model(combined_embeddings, name, train=False, noise_sigma=0.0):
			"""Model to train"""

			if noise_sigma < 0.0:
				raise ValueError("noise sigma must be non-negative {0}: ".format(noise_sigma))

			if train and noise_sigma > 0.0:
				combined_embeddings = combined_embeddings + tf.random_normal(tf.shape(combined_embeddings)) * noise_sigma

			batched_input = tf.expand_dims(combined_embeddings, 0)

			options = {
				"dtype": tf.float32,
				"sequence_length": tf.expand_dims(trans_len, 0),
				"scope": name
			}

			(outputs_fw, outputs_bw), output_states = tf.nn.bidirectional_dynamic_rnn(fw_network, bw_network, batched_input, **options)

			# Add Noise and Predict
			concat_layer = tf.concat(2, [outputs_fw, tf.reverse(outputs_bw, [False, True, False])], name="concat_layer")
			
			if train and noise_sigma > 0.0:
				concat_layer = concat_layer + tf.random_normal(tf.shape(concat_layer)) * noise_sigma

			prediction = tf.log(tf.nn.softmax(tf.matmul(tf.squeeze(concat_layer), weight) + bias), name=name)

			return prediction

		network = model(combined_embeddings, "training", train=True, noise_sigma=config["noise_sigma"])
		trained_model = model(combined_embeddings, "trained")

		# Calculate Loss and Optimize
		labels = tf.placeholder(tf.float32, shape=[None, len(config["tag_map"].keys())], name="y")
		loss = tf.neg(tf.reduce_mean(tf.reduce_sum(network * labels, 1)), name="loss")
		optimizer = tf.train.GradientDescentOptimizer(config["learning_rate"]).minimize(loss, name="optimizer")

		saver = tf.train.Saver()

	return graph, saver

def train_model(config, graph, sess, saver):
	"""Train the model"""

	eras = config["eras"]
	dataset = config["dataset"]
	train = config["train"]
	test = config["test"]
	train_index = list(train.index)
	sess.run(get_op(graph, "assign_wembedding"), feed_dict={get_tensor(graph, "embedding_placeholder:0"): config["wembedding"]})

	# Train the Model
	for step in range(eras):
		random.shuffle(train_index)
		count = 0
		total_loss = 0

		print("ERA: " + str(step))
		np.set_printoptions(threshold=np.inf)

		for t_index in train_index[0:1000]:

			count += 1

			row = train.loc[t_index]
			tokens, tags = get_tags(row)
			char_inputs, word_lengths, word_indices, labels = trans_to_tensor(config, sess, graph, tokens, tags, train=True)

			feed_dict = {
				get_tensor(graph, "char_inputs:0") : char_inputs,
				get_tensor(graph, "word_inputs:0") : word_indices,
				get_tensor(graph, "word_lengths:0") : word_lengths,
				get_tensor(graph, "trans_length:0"): len(tokens),
				get_tensor(graph, "y:0"): labels
			}

			# Run Training Step
			optimizer_out, loss = sess.run([get_op(graph, "optimizer"), get_tensor(graph, "loss:0")], feed_dict=feed_dict)
			total_loss += loss

			# Log
			if count % 500 == 0:
				print(sess.run(get_tensor(graph, "combined_embeddings:0"), feed_dict=feed_dict).shape)
				print("loss: " + str(total_loss / count))
				print("%d" % (count / len(train_index) * 100) + "% complete with era")

		# Evaluate Model
		test_accuracy = evaluate_testset(config, graph, sess, test)

	final_model_path = ""

	return final_model_path

def evaluate_testset(config, graph, sess, test):
	"""Check error on test set"""

	total_count = 0
	total_correct = 0
	test_index = list(test.index)
	random.shuffle(test_index)
	count = 0

	print("---ENTERING EVALUATION---")

	for i in test_index[0:1000]:

		count += 1

		if count % 500 == 0:
			print("%d" % (count / len(test_index) * 100) + "% complete with evaluation")

		row = test.loc[i]
		tokens, tags = get_tags(row)
		char_inputs, word_lengths, word_indices, labels = trans_to_tensor(config, sess, graph, tokens, tags)
		total_count += len(tokens)
		
		feed_dict = {
			get_tensor(graph, "char_inputs:0") : char_inputs,
			get_tensor(graph, "word_inputs:0") : word_indices,
			get_tensor(graph, "word_lengths:0") : word_lengths,
			get_tensor(graph, "trans_length:0"): len(tokens)
		}

		output = sess.run(get_tensor(graph, "trained:0"), feed_dict=feed_dict)

		correct_count = np.sum(np.argmax(output, 1) == np.argmax(labels, 1))
		total_correct += correct_count

	test_accuracy = 100.0 * (total_correct / total_count)
	logging.info("Test accuracy: %.2f%%" % test_accuracy)
	logging.info("Correct count: " + str(total_correct))
	logging.info("Total count: " + str(total_count))

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
