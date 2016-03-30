#!/usr/local/bin/python3
# pylint: disable=unused-variable
# pylint: disable=too-many-locals

"""Train a CNN using tensorFlow

Created on Mar 14, 2016
@author: Matthew Sevrens
@author: Tina Wu
@author: J. Andrew Key
"""

#################### USAGE ########################################################################

# usage: meerkat.classification.tensorflow_cnn [-h] [-d] [-v] config_file

# positional arguments:
#   config_file  What is the location of your configuration file?

# optional arguments:
#   -h, --help   show this help message and exit
#   -d, --debug  log at DEBUG level
#   -v, --info   log at INFO level

# For addtional details on implementation see:
# Character-level Convolutional Networks for Text Classification
# http://arxiv.org/abs/1509.01626

###################################################################################################

import argparse
import csv
import logging
import math
import os
import pprint
import random
import shutil
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from jsonschema import validate

from meerkat.classification.tools import fill_description_unmasked, reverse_map
from meerkat.various_tools import load_params, load_piped_dataframe, validate_configuration

def chunks(array, num):
	"""Chunk array into equal sized parts"""
	num = max(1, num)
	return [array[i:i + num] for i in range(0, len(array), num)]

def parse_arguments():
	"""This function parses arguments from our command line."""
	parser = argparse.ArgumentParser("meerkat.classification.tensorflow_cnn")
	# Required arugments
	parser.add_argument("config_file",
		help="What is the location of your configuration file?")
	parser.add_argument("-d", "--debug", help="log at DEBUG level",
		action="store_true")
	parser.add_argument("-v", "--info", help="log at INFO level",
		action="store_true")

	args = parser.parse_args()
	if args.debug:
		logging.basicConfig(level=logging.DEBUG)
	elif args.info:
		logging.basicConfig(level=logging.INFO)
	logging.info("Arguments parsed.")
	return args

def validate_config(config):
	"""Validate input configuration"""

	config = load_params(config)
	schema_file = "meerkat/classification/config/tensorflow_cnn_schema.json"
	config = validate_configuration(config, schema_file)
	logging.debug("Configuration is :\n{0}".format(pprint.pformat(config)))
	reshape = ((config["doc_length"] - 96) / 27) * 256
	config["label_map"] = load_params(config["label_map"])
	config["num_labels"] = len(config["label_map"].keys())
	config["alpha_dict"] = {a : i for i, a in enumerate(config["alphabet"])}
	config["base_rate"] = config["base_rate"] * math.sqrt(config["batch_size"]) / math.sqrt(128)
	config["alphabet_length"] = len(config["alphabet"])

	if reshape.is_integer():
		config["reshape"] = int(reshape)
	else:
		raise ValueError('DOC_LENGTH - 96 must be divisible by 27: 123, 150, 177, 204...')

	return config

def load_data(config):
	"""Load data and label map"""

	model_type = config["model_type"]
	dataset = config["dataset"]
	label_map = config["label_map"]
	ledger_entry = config["ledger_entry"]

	ground_truth_labels = {
		'merchant' : 'MERCHANT_NAME',
		'subtype' : 'PROPOSED_SUBTYPE'
	}

	label_key = ground_truth_labels[model_type]
	reversed_map = reverse_map(label_map)
	map_labels = lambda x: reversed_map.get(str(x[label_key]), "")

	df = load_piped_dataframe(dataset)

	# Verify number of labels
	if not len(reversed_map) == len(df[label_key].value_counts()):
		logging.critical("Reversed Map :\n{0}".format(pprint.pformat(reversed_map)))
		logging.critical("df[label_key].value_counts(): {0}".format(df[label_key].value_counts()))
		map_keys = reversed_map.keys()
		keys_in_dataframe = df[label_key].value_counts().index.get_values()
		missing_keys = map_keys - keys_in_dataframe
		logging.critical("The dataframe label counts index is missing these {0} items:\n{1}"\
			.format(len(missing_keys), pprint.pformat(missing_keys)))

		raise Exception('Number of indexes does not match number of labels')

	if model_type == "subtype":
		df['LEDGER_ENTRY'] = df['LEDGER_ENTRY'].str.lower()
		grouped = df.groupby('LEDGER_ENTRY', as_index=False)
		groups = dict(list(grouped))
		df = groups[ledger_entry]

	df["DESCRIPTION_UNMASKED"] = df.apply(fill_description_unmasked, axis=1)
	df = df.reindex(np.random.permutation(df.index))
	df["LABEL_NUM"] = df.apply(map_labels, axis=1)
	df = df[df["LABEL_NUM"] != ""]

	msk = np.random.rand(len(df)) < 0.90
	train = df[msk]
	test = df[~msk]

	grouped_train = train.groupby('LABEL_NUM', as_index=False)
	groups_train = dict(list(grouped_train))

	chunked_test = chunks(np.array(test.index), 128)
	return train, test, groups_train, chunked_test

def evaluate_testset(config, graph, sess, model, test, chunked_test):
	"""Check error on test set"""

	total_count = 0
	correct_count = 0
	num_chunks = len(chunked_test)

	for i in range(num_chunks):

		batch_test = test.loc[chunked_test[i]]
		batch_size = len(batch_test)

		trans_test, labels_test = batch_to_tensor(config, batch_test)
		feed_dict_test = {get_tensor(graph, "x:0"): trans_test}
		output = sess.run(model, feed_dict=feed_dict_test)

		batch_correct_count = np.sum(np.argmax(output, 1) == np.argmax(labels_test, 1))

		correct_count += batch_correct_count
		total_count += batch_size
	
	test_accuracy = 100.0 * (correct_count / total_count)
	logging.info("Test accuracy: %.2f%%" % test_accuracy)
	logging.info("Correct count: " + str(correct_count))

	return test_accuracy

def mixed_batching(config, df, groups_train):
	"""Batch from train data using equal class batching"""

	num_labels = config["num_labels"]
	batch_size = config["batch_size"]
	half_batch = int(batch_size / 2)
	indices_to_sample = list(np.random.choice(df.index, half_batch))

	for index in range(half_batch):
		label = random.randint(1, num_labels)
		select_group = groups_train[str(label)]
		indices_to_sample.append(np.random.choice(select_group.index, 1)[0])

	random.shuffle(indices_to_sample)
	batch = df.loc[indices_to_sample]

	return batch

def batch_to_tensor(config, batch):
	"""Convert a batch to a tensor representation"""

	doc_length = config["doc_length"]
	alphabet_length = config["alphabet_length"]
	num_labels = config["num_labels"]
	batch_size = config["batch_size"]

	labels = np.array(batch["LABEL_NUM"].astype(int)) - 1
	labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
	docs = batch["DESCRIPTION_UNMASKED"].tolist()
	transactions = np.zeros(shape=(batch_size, 1, alphabet_length, doc_length))
	
	for index, trans in enumerate(docs):
		transactions[index][0] = string_to_tensor(config, trans, doc_length)

	transactions = np.transpose(transactions, (0, 1, 3, 2))
	return transactions, labels

def string_to_tensor(config, doc, length):
	"""Convert transaction to tensor format"""
	alphabet = config["alphabet"]
	alpha_dict = config["alpha_dict"]
	doc = doc.lower()[0:length]
	tensor = np.zeros((len(alphabet), length), dtype=np.float32)
	for index, char in reversed(list(enumerate(doc))):
		if char in alphabet:
			tensor[alpha_dict[char]][len(doc) - index - 1] = 1
	return tensor
	
def get_tensor(graph, name):
	"""Get tensor by name"""
	return graph.get_tensor_by_name(name)

def get_op(graph, name):
	"""Get operation by name"""
	return graph.get_operation_by_name(name)

def get_variable(graph, name):
	"""Get variable by name"""
	with graph.as_default():
		variable = [v for v in tf.all_variables() if v.name == name][0]
		return variable

def accuracy(predictions, labels):
	"""Return accuracy for a batch"""
	return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

def threshold(tensor):
	"""ReLU with threshold at 1e-6"""
	return tf.mul(tf.to_float(tf.greater_equal(tensor, 1e-6)), tensor)

def softmax_with_temperature(tensor, temperature):
	"""Softmax with temperature variable"""
	return tf.div(tf.exp(tensor/temperature), tf.reduce_sum(tf.exp(tensor/temperature)))

def bias_variable(shape, mult):
	"""Initialize biases"""
	stdv = 1 / math.sqrt(mult)
	bias = tf.Variable(tf.random_uniform(shape, minval=-stdv, maxval=stdv), name="B")
	return bias

def weight_variable(config, shape):
	"""Initialize weights"""
	weight = tf.Variable(tf.mul(tf.random_normal(shape), config["randomize"]), name="W")
	return weight

def conv2d(input_x, weights):
	"""Create convolutional layer"""
	layer = tf.nn.conv2d(input_x, weights, strides=[1, 1, 1, 1], padding='VALID')
	return layer

def max_pool(tensor):
	"""Create max pooling layer"""
	layer = tf.nn.max_pool(tensor, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding='VALID')
	return layer

def build_graph(config):
	"""Build CNN"""

	doc_length = config["doc_length"]
	alphabet_length = config["alphabet_length"]
	reshape = config["reshape"]
	num_labels = config["num_labels"]
	base_rate = config["base_rate"]
	graph = tf.Graph()

	# Create Graph
	with graph.as_default():

		learning_rate = tf.Variable(base_rate, trainable=False, name="lr") 

		input_shape = [None, 1, doc_length, alphabet_length]
		output_shape = [None, num_labels]

		trans_placeholder = tf.placeholder(tf.float32, shape=input_shape, name="x")
		labels_placeholder = tf.placeholder(tf.float32, shape=output_shape, name="y")

		w_conv1 = weight_variable(config, [1, 7, alphabet_length, 256])
		b_conv1 = bias_variable([256], 7 * alphabet_length)

		w_conv2 = weight_variable(config, [1, 7, 256, 256])
		b_conv2 = bias_variable([256], 7 * 256)

		w_conv3 = weight_variable(config, [1, 3, 256, 256])
		b_conv3 = bias_variable([256], 3 * 256)

		w_conv4 = weight_variable(config, [1, 3, 256, 256])
		b_conv4 = bias_variable([256], 3 * 256)

		w_conv5 = weight_variable(config, [1, 3, 256, 256])
		b_conv5 = bias_variable([256], 3 * 256)

		w_fc1 = weight_variable(config, [reshape, 1024])
		b_fc1 = bias_variable([1024], reshape)

		w_fc2 = weight_variable(config, [1024, num_labels])
		b_fc2 = bias_variable([num_labels], 1024)

		def model(data, name, train=False):
			"""Add model layers to the graph"""

			h_conv1 = threshold(conv2d(data, w_conv1) + b_conv1)
			h_pool1 = max_pool(h_conv1)

			h_conv2 = threshold(conv2d(h_pool1, w_conv2) + b_conv2)
			h_pool2 = max_pool(h_conv2)

			h_conv3 = threshold(conv2d(h_pool2, w_conv3) + b_conv3)

			h_conv4 = threshold(conv2d(h_conv3, w_conv4) + b_conv4)

			h_conv5 = threshold(conv2d(h_conv4, w_conv5) + b_conv5)
			h_pool5 = max_pool(h_conv5)

			h_reshape = tf.reshape(h_pool5, [-1, reshape])

			h_fc1 = threshold(tf.matmul(h_reshape, w_fc1) + b_fc1)

			if train:
				h_fc1 = tf.nn.dropout(h_fc1, 0.5)

			h_fc2 = tf.matmul(h_fc1, w_fc2) + b_fc2

			softmax = tf.nn.softmax(h_fc2)
			network = tf.log(tf.clip_by_value(softmax, 1e-10, 1.0), name=name)

			return network

		network = model(trans_placeholder, "network", train=True)
		trained_model = model(trans_placeholder, "model", train=False)

		loss = tf.neg(tf.reduce_mean(tf.reduce_sum(network * labels_placeholder, 1)), name="loss")
		optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, name="optimizer")

		saver = tf.train.Saver()

	return graph, saver

def train_model(config, graph, sess, saver):
	"""Train the model"""

	epochs = config["epochs"]
	eras = config["eras"]
	dataset = config["dataset"]
	train, test, groups_train, chunked_test = load_data(config)
	num_eras = epochs * eras
	logging_interval = 50
	learning_rate_interval = 15000

	best_accuracy, best_era = 0, 0
	save_dir = "meerkat/classification/models/checkpoints/"
	os.makedirs(save_dir, exist_ok=True)
	checkpoints = {}

	for step in range(num_eras):

		# Prepare Data for Training
		batch = mixed_batching(config, train, groups_train)
		trans, labels = batch_to_tensor(config, batch)
		feed_dict = {get_tensor(graph, "x:0") : trans, get_tensor(graph, "y:0") : labels}

		# Run Training Step
		sess.run(get_op(graph, "optimizer"), feed_dict=feed_dict)

		# Log Loss
		if step % logging_interval == 0:
			loss = sess.run(get_tensor(graph, "loss:0"), feed_dict=feed_dict)
			logging.info("train loss at epoch %d: %g" % (step + 1, loss))

		# Evaluate Testset, Log Progress and Save
		if step != 0 and step % epochs == 0:

			#Evaluate Model
			model = get_tensor(graph, "model:0")
			learning_rate = get_variable(graph, "lr:0")
			predictions = sess.run(model, feed_dict=feed_dict)
			logging.info("Testing for era %d" % (step / epochs))
			logging.info("Learning rate at epoch %d: %g" % (step + 1, sess.run(learning_rate)))
			logging.info("Minibatch accuracy: %.1f%%" % accuracy(predictions, labels))
			test_accuracy = evaluate_testset(config, graph, sess, model, test, chunked_test)

			# Save Checkpoint
			current_era = int(step / epochs)
			save_path = saver.save(sess, save_dir + "era_" + str(current_era) + ".ckpt")
			logging.info("Checkpoint saved in file: %s" % save_path)
			checkpoints[current_era] = save_path

			# Stop Training if Converged
			if test_accuracy > best_accuracy:
				best_era = current_era
				best_accuracy = test_accuracy

			if current_era - best_era == 3:
				save_path = checkpoints[best_era]
				break

		# Update Learning Rate
		if step != 0 and step % learning_rate_interval == 0:
			learning_rate = get_variable(graph, "lr:0")
			sess.run(learning_rate.assign(learning_rate / 2))

	# Clean Up Directory
	final_model_path = "meerkat/classification/models/" +\
		os.path.basename(dataset).split(".")[0] + ".ckpt"
	os.rename(save_path, final_model_path)
	shutil.rmtree(save_dir)

	return final_model_path

def run_session(config, graph, saver):
	"""Run Session"""

	with tf.Session(graph=graph) as sess:

		mode = config["mode"]
		model_path = config["model_path"]

		tf.initialize_all_variables().run()

		if mode == "train":
			train_model(config, graph, sess, saver)
		elif mode == "test":
			saver.restore(sess, model_path)
			model = get_tensor(graph, "model:0")
			_, test, _, chunked_test = load_data(config)
			evaluate_testset(config, graph, sess, model, test, chunked_test)

def run_from_command_line():
	"""Run module from command line"""
	args = parse_arguments()
	config = validate_config(args.config_file)
	graph, saver = build_graph(config)
	run_session(config, graph, saver)

if __name__ == "__main__":
	run_from_command_line()
