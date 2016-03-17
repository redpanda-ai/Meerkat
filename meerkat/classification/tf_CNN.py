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
RESHAPE = int(((DOC_LENGTH - 96) / 27) * 256)
ALPHABET_LENGTH = len(ALPHABET)

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

		def model(data, train=False):

			#TODO: ReLU threshold
			h_conv1 = tf.nn.relu(tf.nn.conv2d(data, W_conv1, [1,1,1,1], padding="VALID") + b_conv1)
			h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding='VALID')

			h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, [1,1,1,1], padding="VALID") + b_conv2)
			h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding='VALID')

			h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, [1,1,1,1], padding="VALID") + b_conv3)

			h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, [1,1,1,1], padding="VALID") + b_conv4)

			h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, [1,1,1,1], padding="VALID") + b_conv5)
			h_pool5 = tf.nn.max_pool(h_conv5, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding='VALID')

			h_reshape = tf.reshape(h_pool5, [BATCH_SIZE, RESHAPE])

			h_fc1 = tf.nn.relu(tf.matmul(h_reshape, W_fc1) + b_fc1)

			if train:
				h_fc1 = tf.nn.dropout(h_fc1, 0.5)

			return h_fc1

		network = model(x, train=True)

	def run_session(graph):
		"""Run Session"""

		# Train Network
		label_map, batched = load_data()
		epochs = 5000
		eras = 10

		with tf.Session(graph=graph) as session:

			tf.initialize_all_variables().run()
			num_eras = epochs * eras

			for step in range(num_eras):

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

				print(session.run(network, feed_dict=feed_dict).shape)

				if (step % epochs == 0):

					print("Save details")

				sys.exit()

	# Run Graph
	run_session(graph)

if __name__ == "__main__":
	build_cnn()
