#!/usr/local/bin/python3

"""Train a CNN using tensorFlow

Created on Mar 14, 2016
@author: Matthew Sevrens
@author: Tina Wu
"""

#################### USAGE #######################

# python3 -m meerkat.classification.tf_CNN [config]
# python3 -m meerkat.classification.tf_CNN config/tf_cnn_config.json

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
BASE_RATE = CONFIG["base_rate"] * math.sqrt(BATCH_SIZE) / math.sqrt(128)
DECAY = CONFIG["decay"]
RESHAPE = ((DOC_LENGTH - 96) / 27) * 256
ALPHABET_LENGTH = len(ALPHABET)
EPOCHS = CONFIG["epochs"]
ERAS = CONFIG["eras"]

def chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]

def validate_config():
	"""Validate input configuration"""

	global RESHAPE

	if RESHAPE.is_integer():
		RESHAPE = int(RESHAPE)
	else:
		raise ValueError('DOC_LENGTH - 96 must be divisible by 27: 123, 150, 177, 204...')

def load_data():
	"""Load data and label map"""

	reversed_map = reverse_map(LABEL_MAP)
	a = lambda x: reversed_map.get(str(x["PROPOSED_SUBTYPE"]), "")

	df = pd.read_csv(DATASET, quoting=csv.QUOTE_NONE, na_filter=False, encoding="utf-8", sep='|', error_bad_lines=False)

	df['LEDGER_ENTRY'] = df['LEDGER_ENTRY'].str.lower()
	grouped = df.groupby('LEDGER_ENTRY', as_index=False)
	groups = dict(list(grouped))
	df = groups[LEDGER_ENTRY]
	df["DESCRIPTION_UNMASKED"] = df.apply(fill_description_unmasked, axis=1)
	df = df.reindex(np.random.permutation(df.index))
	df["LABEL_NUM"] = df.apply(a, axis=1)
	df = df[df["LABEL_NUM"] != ""]

	msk = np.random.rand(len(df)) < 0.90
	train = df[msk]
	test = df[~msk]

	grouped_train = train.groupby('LABEL_NUM', as_index=False)
	groups_train = dict(list(grouped_train))

	chunked_test = chunks(np.array(test.index), 128)
	return train, test, groups_train, chunked_test

def evaluate_testset(graph, test, chunked_test, no_dropout, session):
	"""Check error on test set"""

	total_count = 0
	correct_count = 0

	for i in range(len(chunked_test)):

		batch_test = test.loc[chunked_test[i]]
		batch_length = len(batch_test)
		if batch_length != 128: continue

		trans_test, labels_test = batch_to_tensor(batch_test)
		feed_dict_test = {get_tensor(graph, "x:0"): trans_test}
		output = session.run(no_dropout, feed_dict=feed_dict_test)

		batch_correct_count = np.sum(np.argmax(output, 1) == np.argmax(labels_test, 1))

		correct_count += batch_correct_count
		total_count += BATCH_SIZE
	
	test_accuracy = 100.0 * (correct_count / total_count)
	print("Test accuracy: %.2f%%" % test_accuracy)
	print("Correct count: " + str(correct_count))

def mixed_batching(df, groups_train):
	"""Batch from train data using equal class batching"""
	half_batch = int(BATCH_SIZE / 2)
	indices_to_sample = list(np.random.choice(df.index, half_batch))
	for i in range(half_batch):
		label = random.randint(1, NUM_LABELS)
		select_group = groups_train[str(label)]
		indices_to_sample.append(np.random.choice(select_group.index, 1)[0])
	random.shuffle(indices_to_sample)
	batch = df.loc[indices_to_sample]
	return batch

def batch_to_tensor(batch):
	"""Convert a batch to a tensor representation"""

	labels = np.array(batch["LABEL_NUM"].astype(int)) - 1
	labels = (np.arange(NUM_LABELS) == labels[:,None]).astype(np.float32)
	docs = batch["DESCRIPTION_UNMASKED"].tolist()
	trans = np.zeros(shape=(BATCH_SIZE, 1, ALPHABET_LENGTH, DOC_LENGTH))
	
	for i, t in enumerate(docs):
		trans[i][0] = string_to_tensor(t, DOC_LENGTH)

	trans = np.transpose(trans, (0, 1, 3, 2))
	return trans, labels

def string_to_tensor(doc, l):
	"""Convert transaction to tensor format"""
	s = doc.lower()[0:l]
	t = np.zeros((len(ALPHABET), l), dtype=np.float32)
	for i, c in reversed(list(enumerate(s))):
		if c in ALPHABET:
			t[ALPHA_DICT[c]][len(s) - i - 1] = 1
	return t
	
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
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def threshold(tensor):
	return tf.mul(tf.to_float(tf.greater_equal(tensor, 1e-6)), tensor)

def softmax_with_temperature(tensor, T):
	"""Softmax with temperature variable"""
	return tf.div(tf.exp(tensor/T), tf.reduce_sum(tf.exp(tensor/T)))

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

def build_graph():
	"""Build CNN"""

	graph = tf.Graph()

	# Create Graph
	with graph.as_default():

		learning_rate = tf.Variable(BASE_RATE, trainable=False, name="lr") 

		x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 1, DOC_LENGTH, ALPHABET_LENGTH], name="x")
		y = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS), name="y")
		
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

		W_fc1 = weight_variable([RESHAPE, 1024])
		b_fc1 = bias_variable([1024], RESHAPE)

		W_fc2 = weight_variable([1024, NUM_LABELS])
		b_fc2 = bias_variable([NUM_LABELS], 1024)

		def model(data, name, train=False):

			h_conv1 = threshold(conv2d(data, W_conv1) + b_conv1)
			h_pool1 = max_pool(h_conv1)

			h_conv2 = threshold(conv2d(h_pool1, W_conv2) + b_conv2)
			h_pool2 = max_pool(h_conv2)

			h_conv3 = threshold(conv2d(h_pool2, W_conv3) + b_conv3)

			h_conv4 = threshold(conv2d(h_conv3, W_conv4) + b_conv4)

			h_conv5 = threshold(conv2d(h_conv4, W_conv5) + b_conv5)
			h_pool5 = max_pool(h_conv5)

			h_reshape = tf.reshape(h_pool5, [BATCH_SIZE, RESHAPE])

			h_fc1 = threshold(tf.matmul(h_reshape, W_fc1) + b_fc1)

			if train:
				h_fc1 = tf.nn.dropout(h_fc1, 0.5)

			h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

			softmax = tf.nn.softmax(h_fc2)
			network = tf.log(tf.clip_by_value(softmax, 1e-10, 1.0), name=name)

			return network

		network = model(x, "network", train=True)
		trained_model = model(x, "model", train=False)

		loss = tf.neg(tf.reduce_mean(tf.reduce_sum(network * y, 1)), name="loss")
		optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, name="optimizer")

		saver = tf.train.Saver()

	return graph, saver

def train_model(sess, train, test, groups_train, chunked_test):
	"""Train the model"""

	num_eras = EPOCHS * ERAS

	for step in range(num_eras):

		# Prepare Data for Training
		batch = mixed_batching(train, groups_train)
		trans, labels = batch_to_tensor(batch)
		feed_dict = {get_tensor(graph, "x:0") : trans, get_tensor(graph, "y:0") : labels}

		# Run Training Step
		sess.run(get_op(graph, "optimizer"), feed_dict=feed_dict)

		# Log Loss
		if (step % 50 == 0):
			print("train loss at epoch %d: %g" % (step + 1, sess.run(get_tensor(graph, "loss:0"), feed_dict=feed_dict)))

		# Evaluate Testset and Log Progress
		if (step != 0 and step % EPOCHS == 0):
			model = get_tensor(graph, "model:0")
			lr = get_variable(graph, "lr:0")
			predictions = sess.run(model, feed_dict=feed_dict)
			print("Testing for era %d" % (step / EPOCHS))
			print("Learning rate at epoch %d: %g" % (step + 1, sess.run(lr)))
			print("Minibatch accuracy: %.1f%%" % accuracy(predictions, labels))
			evaluate_testset(graph, test, chunked_test, model, sess)

		# Update Learning Rate
		if (step != 0 and step % 15000 == 0):
			lr = get_variable(graph, "lr:0")
			sess.run(lr.assign(lr / 2))

	# Save Model  
	save_path = saver.save(sess, "meerkat/classification/models/model_" + DATASET.split(".")[0] + ".ckpt")
	print("Model saved in file: %s" % save_path)

def run_session(graph, saver):
	"""Run Session"""

	train, test, groups_train, chunked_test = load_data()

	with tf.Session(graph=graph) as sess:

		tf.initialize_all_variables().run()

		if MODE == "train":
			train_model(sess, train, test, groups_train, chunked_test)
		elif MODE == "test":
			saver.restore(sess, MODEL)
			model = get_tensor(graph, "model:0")
			evaluate_testset(graph, test, chunked_test, model, sess)

if __name__ == "__main__":
	validate_config()
	graph, saver = build_graph()
	run_session(graph, saver)