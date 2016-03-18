#!/usr/local/bin/python3

"""Train a CNN using tensorFlow

Created on Mar 14, 2016
@author: Matthew Sevrens
@author: Tina Wu
"""

#################### USAGE #######################

# python3 -m meerkat.classification.tf_CNN

##################################################

import tensorflow as tf
import numpy as np
import pandas as pd

import sys
import csv
import json
import math
import random
from math import sqrt
from .verify_data import load_json
from .tools import fill_description_unmasked, reverse_map

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
ALPHA_DICT = {a : i for i, a in enumerate(ALPHABET)}
NUM_LABELS = 10
BATCH_SIZE = 128
DOC_LENGTH = 123
RANDOMIZE = 5e-2
RESHAPE = ((DOC_LENGTH - 96) / 27) * 256
ALPHABET_LENGTH = len(ALPHABET)

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def load_data():
	"""Load data and label map"""

	label_map = "card_credit_subtype_label_map.json"
	label_map = load_json(label_map)
	reversed_map = reverse_map(label_map)
	a = lambda x: reversed_map.get(str(x["PROPOSED_SUBTYPE"]), "")

	input_file = "Card_complete_data_subtype_original_updated_credit.csv"
	df = pd.read_csv(input_file, quoting=csv.QUOTE_NONE, na_filter=False, encoding="utf-8", sep='|', error_bad_lines=False)

	df['LEDGER_ENTRY'] = df['LEDGER_ENTRY'].str.lower()
	grouped = df.groupby('LEDGER_ENTRY', as_index=False)
	groups = dict(list(grouped))
	df = groups["credit"]
	df["DESCRIPTION_UNMASKED"] = df.apply(fill_description_unmasked, axis=1)
	df = df.reindex(np.random.permutation(df.index))
	df["LABEL_NUM"] = df.apply(a, axis=1)
	df = df[df["LABEL_NUM"] != ""]

	batched = np.array_split(df, math.ceil(df.shape[0] / 129))

	return label_map, batched

def validate_config():
	"""Validate input configuration"""

	global RESHAPE

	if RESHAPE.is_integer():
		RESHAPE = int(RESHAPE)
	else:
		raise ValueError('DOC_LENGTH - 96 must be divisible by 27: 123, 150, 177, 204...')

def string_to_tensor(str, l):
	"""Convert transaction to tensor format"""
	s = str.lower()[0:l]
	t = np.zeros((len(ALPHABET), l), dtype=np.float32)
	for i, c in reversed(list(enumerate(s))):
		if c in ALPHABET:
			t[ALPHA_DICT[c]][len(s) - i - 1] = 1
	return t

def bias_variable(shape, mult):
	"""Initialize biases"""
	stdv = 1 / sqrt(mult)
	bias = tf.Variable(tf.random_uniform(shape, minval=-stdv, maxval=stdv), name="B")
	return bias

def weight_variable(shape):
	"""Initialize weights"""
	weight = tf.Variable(tf.mul(tf.random_normal(shape), RANDOMIZE), name="W")
	return weight

def conv2d(x, W):
	"""Create convolutional layer"""
	layer = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
	return layer

def max_pool(x):
	"""Create max pooling layer"""
	layer = tf.nn.max_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding='VALID')
	return layer

def build_cnn():
	"""Build CNN"""

	graph = tf.Graph()

	# Create Graph
	with graph.as_default():

		x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 1, DOC_LENGTH, ALPHABET_LENGTH])
		y = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
		
		W_conv1 = weight_variable([1, 7, ALPHABET_LENGTH, 256])
		b_conv1 = bias_variable([256], 7 * ALPHABET_LENGTH)

		W_conv2 = weight_variable([1, 7, 256, 256])
		b_conv2 = bias_variable([256], 7 * 256)

		W_conv3 = weight_variable([1, 3, 256, 256])
		b_conv3 = bias_variable([256], 3 * 256)

		W_conv4 = weight_variable([1, 3, 256, 256])
		b_conv4 = bias_variable([256], 3 * 256)

		W_conv5 = weight_variable([1, 3, 256, 256])
		b_conv5 = bias_variable([256], 3 * 256)

		W_conv5 = weight_variable([1, 3, 256, 256])
		b_conv5 = bias_variable([256], 3 * 256)

		W_fc1 = weight_variable([RESHAPE, 1024])
		b_fc1 = bias_variable([1024], RESHAPE)

		W_fc2 = weight_variable([1024, NUM_LABELS])
		b_fc2 = bias_variable([NUM_LABELS], 1024)

		def model(data, train=False):

			#TODO: ReLU threshold
			h_conv1 = tf.nn.relu(conv2d(data, W_conv1) + b_conv1)
			h_pool1 = max_pool(h_conv1)

			h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
			h_pool2 = max_pool(h_conv2)

			h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

			h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

			h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)
			h_pool5 = max_pool(h_conv5)

			h_reshape = tf.reshape(h_pool5, [BATCH_SIZE, RESHAPE])

			h_fc1 = tf.nn.relu(tf.matmul(h_reshape, W_fc1) + b_fc1)

			if train:
				h_fc1 = tf.nn.dropout(h_fc1, 0.5)

			h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

			network = tf.log(tf.nn.softmax(h_fc2))

			return network

		network = model(x, train=True)
		no_dropout = model(x, train=False)

		loss = -tf.reduce_mean(tf.reduce_sum(network * y, 1))
		global_step = tf.Variable(0, trainable=False)
		learning_rate = tf.train.exponential_decay(0.01, global_step, 2000, 0.95, staircase=True) # TODO ensure correct learning rates 
		optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=global_step)

	def run_session(graph):
		"""Run Session"""

		# Train Network
		label_map, batched = load_data()
		epochs = 500
		eras = 10

		with tf.Session(graph=graph) as session:

			tf.initialize_all_variables().run()
			num_eras = epochs * eras

			for step in range(50000):

				batch = random.choice(batched)[0:128]
				labels = np.array(batch["LABEL_NUM"].astype(int))
				labels = (np.arange(NUM_LABELS) == labels[:,None]).astype(np.float32)
				docs = batch["DESCRIPTION_UNMASKED"].tolist()
				trans = np.zeros(shape=(BATCH_SIZE, 1, ALPHABET_LENGTH, DOC_LENGTH))
				for i, t in enumerate(docs):
					trans[i][0] = string_to_tensor(t, DOC_LENGTH)
				# TODO explore need for transpose
				trans = np.transpose(trans, (0, 1, 3, 2))

				feed_dict = {x : trans, y : labels}

				_, l, predictions = session.run([optimizer, loss, network], feed_dict=feed_dict)

				if (step % 50 == 0):
					print("train loss %g"%session.run(loss, feed_dict=feed_dict))

				if (step % epochs == 0):
					print("No dropout accuracy: %.1f%%" % accuracy(session.run(no_dropout, feed_dict=feed_dict), labels))
					print("Minibatch accuracy: %.1f%%" % accuracy(predictions, labels))

	# Run Graph
	run_session(graph)

if __name__ == "__main__":
	validate_config()
	build_cnn()
