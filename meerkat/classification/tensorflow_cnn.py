#!/usr/local/bin/python3

"""Train a CNN using tensorFlow

Created on Mar 14, 2016
@author: Matthew Sevrens
@author: Tina Wu
"""

#################### USAGE #######################

# python3 -m meerkat.classification.tensorflow_cnn [config]
# python3 -m meerkat.classification.tensorflow_cnn config/tf_cnn_config.json

##################################################

import csv
import logging
import math
import numpy as np
import pandas as pd
import random
import sys
import tensorflow as tf

from .tools import fill_description_unmasked, reverse_map
from .verify_data import load_json

#MAGIC NUMBERS
MAGIC_NUMBER_1 = 96 #From the source literature, unsure why
MAGIC_NUMBER_2 = 27 #From the source literature, unsure why
SMALL_FRAME_SIZE = 256

CONFIG = load_json(sys.argv[1])
DATASET = CONFIG["dataset"]
MODE = CONFIG["mode"]
MODEL = CONFIG["model"]
MODEL_TYPE = CONFIG["model_type"]
CONTAINER = CONFIG["container"]
LEDGER_ENTRY = CONFIG["ledger_entry"]
ALPHABET = CONFIG["alphabet"]
ALPHA_DICT = {a : i for i, a in enumerate(ALPHABET)}
LABEL_MAP = load_json(CONFIG["label_map"])
NUM_LABELS = len(LABEL_MAP.keys())
BATCH_SIZE = CONFIG["batch_size"]
DOC_LENGTH = CONFIG["doc_length"]
RANDOMIZE = CONFIG["randomize"]
MOMENTUM = CONFIG["momentum"]
BASE_RATE = CONFIG["base_rate"] * math.sqrt(BATCH_SIZE) / math.sqrt(BATCH_SIZE)
DECAY = CONFIG["decay"]
RESHAPE = ((DOC_LENGTH - MAGIC_NUMBER_1) / MAGIC_NUMBER_2) * SMALL_FRAME_SIZE
ALPHABET_LENGTH = len(ALPHABET)
EPOCHS = CONFIG["epochs"]
ERAS = CONFIG["eras"]

def chunks(array, num):
	"""Chunk array into equal sized parts"""
	num = max(1, num)
	return [array[i:i + num] for i in range(0, len(array), num)]

def validate_config():
	"""Validate input configuration"""

	global RESHAPE

	if RESHAPE.is_integer():
		RESHAPE = int(RESHAPE)
	else:
		raise ValueError('DOC_LENGTH - MAGIC_NUMBER_1 must be divisible by MAGIC_NUMBER_2: 123,'
			' 150, 177, 204...')

def load_data():
	"""Load data and label map"""

	reversed_map = reverse_map(LABEL_MAP)
	map_labels = lambda x: reversed_map.get(str(x["PROPOSED_SUBTYPE"]), "")

	df = pd.read_csv(DATASET, quoting=csv.QUOTE_NONE, na_filter=False, encoding="utf-8",
		sep='|', error_bad_lines=False)

	df['LEDGER_ENTRY'] = df['LEDGER_ENTRY'].str.lower()
	grouped = df.groupby('LEDGER_ENTRY', as_index=False)
	groups = dict(list(grouped))
	df = groups[LEDGER_ENTRY]
	df["DESCRIPTION_UNMASKED"] = df.apply(fill_description_unmasked, axis=1)
	df = df.reindex(np.random.permutation(df.index))
	df["LABEL_NUM"] = df.apply(map_labels, axis=1)
	df = df[df["LABEL_NUM"] != ""]

	msk = np.random.rand(len(df)) < 0.90
	train = df[msk]
	test = df[~msk]

	grouped_train = train.groupby('LABEL_NUM', as_index=False)
	groups_train = dict(list(grouped_train))

	chunked_test = chunks(np.array(test.index), BATCH_SIZE)
	return train, test, groups_train, chunked_test

def evaluate_testset(graph, sess, model, test, chunked_test):
	"""Check error on test set"""

	total_count = 0
	correct_count = 0

	for i in range(len(chunked_test)):

		batch_test = test.loc[chunked_test[i]]
		batch_length = len(batch_test)

		if batch_length != BATCH_SIZE:
			continue

		trans_test, labels_test = batch_to_tensor(batch_test)
		feed_dict_test = {get_tensor(graph, "x:0"): trans_test}
		output = sess.run(model, feed_dict=feed_dict_test)

		batch_correct_count = np.sum(np.argmax(output, 1) == np.argmax(labels_test, 1))

		correct_count += batch_correct_count
		total_count += BATCH_SIZE
	
	test_accuracy = 100.0 * (correct_count / total_count)
	logging.warning("Test accuracy: %.2f%%" % test_accuracy)
	logging.warning("Correct count: " + str(correct_count))

def mixed_batching(df, groups_train):
	"""Batch from train data using equal class batching"""
	half_batch = int(BATCH_SIZE / 2)
	indices_to_sample = list(np.random.choice(df.index, half_batch))
	for index in range(half_batch):
		label = random.randint(1, NUM_LABELS)
		select_group = groups_train[str(label)]
		indices_to_sample.append(np.random.choice(select_group.index, 1)[0])
	random.shuffle(indices_to_sample)
	batch = df.loc[indices_to_sample]
	return batch

def batch_to_tensor(batch):
	"""Convert a batch to a tensor representation"""

	labels = np.array(batch["LABEL_NUM"].astype(int)) - 1
	labels = (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32)
	docs = batch["DESCRIPTION_UNMASKED"].tolist()
	transactions = np.zeros(shape=(BATCH_SIZE, 1, ALPHABET_LENGTH, DOC_LENGTH))
	
	for index, trans in enumerate(docs):
		transactions[index][0] = string_to_tensor(trans, DOC_LENGTH)

	transactions = np.transpose(transactions, (0, 1, 3, 2))
	return transactions, labels

def string_to_tensor(doc, length):
	"""Convert transaction to tensor format"""
	doc = doc.lower()[0:length]
	tensor = np.zeros((len(ALPHABET), length), dtype=np.float32)
	for index, char in reversed(list(enumerate(doc))):
		if char in ALPHABET:
			tensor[ALPHA_DICT[char]][len(doc) - index - 1] = 1
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

def weight_variable(shape):
	"""Initialize weights"""
	weight = tf.Variable(tf.mul(tf.random_normal(shape), RANDOMIZE), name="W")
	return weight

def conv2d(input_x, weights):
	"""Create convolutional layer"""
	layer = tf.nn.conv2d(input_x, weights, strides=[1, 1, 1, 1], padding='VALID')
	return layer

def max_pool(tensor):
	"""Create max pooling layer"""
	layer = tf.nn.max_pool(tensor, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding='VALID')
	return layer

def build_graph():
	"""Build CNN"""

	graph = tf.Graph()

	# Create Graph
	with graph.as_default():

		learning_rate = tf.Variable(BASE_RATE, trainable=False, name="lr") 

		trans_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 1, DOC_LENGTH, ALPHABET_LENGTH],
			name="x")
		labels_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS),
			name="y")
		# Five convolutional layers
		# First, two layers using a kernel width of 7
		conv_width = 256 # width of convolutional frame
		kern_7 = 7 # our kernel width is 7
		w_conv_1 = weight_variable([1, kern_7, ALPHABET_LENGTH, conv_width])
		b_conv_1 = bias_variable([conv_width], kern_7 * ALPHABET_LENGTH)

		w_conv_2 = weight_variable([1, kern_7, conv_width, conv_width])
		b_conv_2 = bias_variable([conv_width], kern_7 * conv_width)

		# Next, three layers using a kernel width of 3
		kern_3 = 3 # our kernel width is 3
		w_conv_3 = weight_variable([1, kern_3, conv_width, conv_width])
		b_conv_3 = bias_variable([conv_width], kern_3 * conv_width)

		w_conv_4 = weight_variable([1, kern_3, conv_width, conv_width])
		b_conv_4 = bias_variable([conv_width], kern_3 * conv_width)

		w_conv_5 = weight_variable([1, kern_3, conv_width, conv_width])
		b_conv_5 = bias_variable([conv_width], kern_3 * conv_width)

		# Finally, 2 linear (fully connected) layers
		linear_width = 1024 # width of fully-connected (linear) layer
		w_linear_1 = weight_variable([RESHAPE, linear_width])
		b_linear_1 = bias_variable([linear_width], RESHAPE)

		w_linear_2 = weight_variable([linear_width, NUM_LABELS])
		b_linear_2 = bias_variable([NUM_LABELS], linear_width)

		def model(data, name, train=False):
			"""Add model layers to the graph"""

			h_conv_1 = threshold(conv2d(data, w_conv_1) + b_conv_1)
			h_pool_1 = max_pool(h_conv_1)

			h_conv_2 = threshold(conv2d(h_pool_1, w_conv_2) + b_conv_2)
			h_pool_2 = max_pool(h_conv_2)

			h_conv_3 = threshold(conv2d(h_pool_2, w_conv_3) + b_conv_3)

			h_conv_4 = threshold(conv2d(h_conv_3, w_conv_4) + b_conv_4)

			h_conv_5 = threshold(conv2d(h_conv_4, w_conv_5) + b_conv_5)
			h_pool_5 = max_pool(h_conv_5)

			h_reshape = tf.reshape(h_pool_5, [BATCH_SIZE, RESHAPE])

			h_linear_1 = threshold(tf.matmul(h_reshape, w_linear_1) + b_linear_1)

			if train:
				h_linear_1 = tf.nn.dropout(h_linear_1, 0.5)

			h_linear_2 = tf.matmul(h_linear_1, w_linear_2) + b_linear_2

			softmax = tf.nn.softmax(h_linear_2)
			network = tf.log(tf.clip_by_value(softmax, 1e-10, 1.0), name=name)

			return network

		network = model(trans_placeholder, "network", train=True)
		trained_model = model(trans_placeholder, "model", train=False)

		loss = tf.neg(tf.reduce_mean(tf.reduce_sum(network * labels_placeholder, 1)), name="loss")
		optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, name="optimizer")

		saver = tf.train.Saver()

	return graph, saver

def train_model(graph, sess, saver):
	"""Train the model"""

	train, test, groups_train, chunked_test = load_data()
	num_eras = EPOCHS * ERAS

	logging_interval = 50
	learning_rate_interval = 15000
	for step in range(num_eras):

		# Prepare Data for Training
		batch = mixed_batching(train, groups_train)
		trans, labels = batch_to_tensor(batch)
		feed_dict = {get_tensor(graph, "x:0") : trans, get_tensor(graph, "y:0") : labels}

		# Run Training Step
		sess.run(get_op(graph, "optimizer"), feed_dict=feed_dict)

		# Log Loss
		if step % logging_interval == 0:
			logging.warning("train loss at epoch %d: %g" % (step + 1, sess.run(get_tensor(graph,
				"loss:0"), feed_dict=feed_dict)))

		# Evaluate Testset and Log Progress
		if step != 0 and step % EPOCHS == 0:
			model = get_tensor(graph, "model:0")
			learning_rate = get_variable(graph, "lr:0")
			predictions = sess.run(model, feed_dict=feed_dict)
			logging.warning("Testing for era %d" % (step / EPOCHS))
			logging.warning("Learning rate at epoch %d: %g" % (step + 1, sess.run(learning_rate)))
			logging.warning("Minibatch accuracy: %.1f%%" % accuracy(predictions, labels))
			evaluate_testset(graph, sess, model, test, chunked_test)

		# Update Learning Rate
		if step != 0 and step % learning_rate_interval == 0:
			learning_rate = get_variable(graph, "lr:0")
			sess.run(learning_rate.assign(learning_rate / 2))

	# Save Model
	save_path = saver.save(sess,
		"meerkat/classification/models/model_" + DATASET.split(".")[0] + ".ckpt")
	logging.warning("Model saved in file: %s" % save_path)

def run_session(graph, saver):
	"""Run Session"""

	with tf.Session(graph=graph) as sess:

		tf.initialize_all_variables().run()

		if MODE == "train":
			train_model(graph, sess, saver)
		elif MODE == "test":
			saver.restore(sess, MODEL)
			model = get_tensor(graph, "model:0")
			_, test, _, chunked_test = load_data()
			evaluate_testset(graph, sess, model, test, chunked_test)

def run_from_command_line():
	"""Run module from command line"""
	validate_config()
	graph, saver = build_graph()
	run_session(graph, saver)

if __name__ == "__main__":
	run_from_command_line()
