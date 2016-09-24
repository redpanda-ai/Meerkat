#!/usr/local/bin/python3
# pylint: disable=unused-variable
# pylint: disable=unused-argument
# pylint: disable=too-many-locals

"""Train a CNN using tensorFlow

Created on Apr 16, 2016
@author: Oscar Pan
@author: Jie Zhang
"""

############################################# USAGE ###############################################

# meerkat.longtail.cnn_sws [config_file]
# meerkat.longtail.cnn_sws meerkat/longtail/cnn_sws_config.json

# For addtional details on implementation see:
# Character-level Convolutional Networks for Text Classification
# http://arxiv.org/abs/1509.01626
#
# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
# http://arxiv.org/abs/1502.03167

###################################################################################################

import logging
import math
import os
import pprint
import random
import shutil
import sys

import numpy as np
import tensorflow as tf

from meerkat.classification.tools import (fill_description_unmasked, batch_normalization, chunks,
	accuracy, get_tensor, get_op, get_variable, threshold, bias_variable, weight_variable, conv2d,
	max_pool, get_cost_list, string_to_tensor)
from meerkat.various_tools import load_params, load_piped_dataframe, validate_configuration
from meerkat.classification.tensorflow_cnn import (mixed_batching, batch_to_tensor,
	evaluate_testset)

logging.basicConfig(level=logging.INFO)
def validate_config(config):
	"""Validate input configuration"""

	config = load_params(config)
	schema = "meerkat/longtail/sws_schema.json"
	config = validate_configuration(config, schema)
	logging.debug("Configuration is :\n{0}".format(pprint.pformat(config)))
	config["reshape"] = int(((config["doc_length"]-78)/27)*256)
	config["num_labels"] = 2
	config["alpha_dict"] = {a: i for i, a in enumerate(config["alphabet"])}
	config["base_rate"] = config["base_rate"]
	config["alphabet_length"] = len(config["alphabet"])
	return config

def load_data(config):
	"""Load labeled data and label map"""

	df = load_piped_dataframe(config["dataset"])
	map_labels = lambda x: "2" if x[config["label_name"]] == "" else "1"
	df["LABEL_NUM"] = df.apply(map_labels, axis=1)
	msk = np.random.rand(len(df)) < 0.9
	train = df[msk]
	test = df[~msk]
	grouped_train = train.groupby("LABEL_NUM", as_index=False)
	groups_train = dict(list(grouped_train))
	return train, test, groups_train

def build_graph(config):
	"""Build CNN"""

	doc_length = config["doc_length"]
	alphabet_length = config["alphabet_length"]
	reshape = config["reshape"]
	num_labels = config["num_labels"]
	base_rate = config["base_rate"]
	batch_size = config["batch_size"]
	graph = tf.Graph()

	# Create Graph
	with graph.as_default():

		learning_rate = tf.Variable(base_rate, trainable=False, name="lr")
		test_accuracy = tf.Variable(0, trainable=False, name="test_accuracy")
		tf.scalar_summary('test_accuracy', test_accuracy)

					# [batch, height, width, channels]
		input_shape = [None, 1, doc_length, alphabet_length]
		output_shape = [None, num_labels]

		trans_placeholder = tf.placeholder(tf.float32, shape=input_shape, name="x")
		labels_placeholder = tf.placeholder(tf.float32, shape=output_shape, name="y")

		# Encoder Weights and Biases
		w_conv1 = weight_variable(config, [1, 7, alphabet_length, 256]) #  1 x 7 is kernel size
		b_conv1 = bias_variable([256], 7 * alphabet_length)

		w_conv2 = weight_variable(config, [1, 7, 256, 256])
		b_conv2 = bias_variable([256], 7 * 256)

		w_conv3 = weight_variable(config, [1, 3, 256, 256]) # 1 x 3 is kernel size
		b_conv3 = bias_variable([256], 3 * 256)

		w_conv4 = weight_variable(config, [1, 3, 256, 256])
		b_conv4 = bias_variable([256], 3 * 256)

		w_conv5 = weight_variable(config, [1, 3, 256, 256])
		b_conv5 = bias_variable([256], 3 * 256)

		w_fc1 = weight_variable(config, [reshape, 1024])
		b_fc1 = bias_variable([1024], reshape)

		w_fc2 = weight_variable(config, [1024, num_labels])
		b_fc2 = bias_variable([num_labels], 1024)

		# Utility for Batch Normalization
		bn_scaler = tf.Variable(1.0 * tf.ones([num_labels]))
		layer_sizes = [256] * 8 + [1024, num_labels]
		ewma = tf.train.ExponentialMovingAverage(decay=0.99)
		bn_assigns = []

		with tf.name_scope("running_mean"):
			running_mean = [tf.Variable(tf.zeros([l]), trainable=False) for l in layer_sizes]

		with tf.name_scope("running_var"):
			running_var = [tf.Variable(tf.ones([l]), trainable=False) for l in layer_sizes]

		def layer(*args, **kwargs):
			"""Apply all necessary steps in a ladder layer"""
			input_h, details, layer_name, train = args[:]
			weights = kwargs.get('weights', None)
			biases = kwargs.get('biases', None)

			# Scope for Visualization with TensorBoard
			with tf.name_scope(layer_name):

				# Preactivation
				if "conv" in layer_name:
					z_pre = conv2d(input_h, weights)
				elif "pool" in layer_name:
					z_pre = max_pool(input_h)
				elif "fc" in layer_name:
					z_pre = tf.matmul(input_h, weights)

				details["layer_count"] += 1
				layer_n = details["layer_count"]

				if train:
					normalized_layer = update_batch_normalization(z_pre, layer_n)
				else:
					mean = ewma.average(running_mean[layer_n-1])
					var = ewma.average(running_var[layer_n-1])
					normalized_layer = batch_normalization(z_pre, mean=mean, var=var)

				# Apply Activation
				if "conv" in layer_name or "fc" in layer_name:
					layer = threshold(normalized_layer + biases)
				else:
					layer = normalized_layer

			return layer

		def update_batch_normalization(batch, layer_n):
			"batch normalize + update average mean and variance of layer l"
			axes = [0] if len(batch.get_shape()) == 2 else [0, 1, 2]
			mean, var = tf.nn.moments(batch, axes=axes)
			assign_mean = running_mean[layer_n-1].assign(mean)
			assign_var = running_var[layer_n-1].assign(var)
			bn_assigns.append(ewma.apply([running_mean[layer_n-1], running_var[layer_n-1]]))
			with tf.control_dependencies([assign_mean, assign_var]):
				return (batch - mean) / tf.sqrt(var + 1e-10)

		def encoder(inputs, name, train=False, noise_std=0.0):
			"""Add model layers to the graph"""

			details = {"layer_count": 0}

			h_conv1 = layer(inputs, details, "h_conv1", train, weights=w_conv1, biases=b_conv1)
			h_pool1 = layer(h_conv1, details, "h_pool1", train)

			h_conv2 = layer(h_pool1, details, "h_conv2", train, weights=w_conv2, biases=b_conv2)
			h_pool2 = layer(h_conv2, details, "h_pool2", train)

			h_conv3 = layer(h_pool2, details, "h_conv3", train, weights=w_conv3, biases=b_conv3)

			h_conv4 = layer(h_conv3, details, "h_conv4", train, weights=w_conv4, biases=b_conv4)

			h_conv5 = layer(h_conv4, details, "h_conv5", train, weights=w_conv5, biases=b_conv5)
			h_pool5 = layer(h_conv5, details, "h_pool5", train)

			h_reshape = tf.reshape(h_pool5, [-1, reshape])

			h_fc1 = layer(h_reshape, details, "h_fc1", train, weights=w_fc1, biases=b_fc1)

			if train:
				h_fc1 = tf.nn.dropout(h_fc1, 0.5)

			h_fc2 = layer(h_fc1, details, "h_fc2", train, weights=w_fc2, biases=b_fc2)

			softmax = tf.nn.softmax(bn_scaler * h_fc2)
			network = tf.log(tf.clip_by_value(softmax, 1e-10, 1.0), name=name)

			return network

		network = encoder(trans_placeholder, "network", train=True)
		_ = encoder(trans_placeholder, "model", train=False)

		# Calculate Loss and Optimize
		with tf.name_scope('trainer'):
			loss = tf.neg(tf.reduce_mean(tf.reduce_sum(network * labels_placeholder, 1)), name="loss")
			optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, name="optimizer")
			tf.scalar_summary('loss', loss)

		bn_updates = tf.group(*bn_assigns)
		with tf.control_dependencies([optimizer]):
			bn_applier = tf.group(bn_updates, name="bn_applier")

		saver = tf.train.Saver()

	return graph, saver

def train_model(config, graph, sess, saver):
	"""Train the model"""

	epochs = config["epochs"]
	eras = config["eras"]
	dataset = config["dataset"]
	train, test, groups_train = load_data(config)
	num_eras = epochs * eras
	logging_interval = 50
	learning_rate_interval = 15000

	best_accuracy, best_era = 0, 0
	base = "meerkat/longtail/sws_model/"
	save_dir = base + "checkpoints/"
	os.makedirs(save_dir, exist_ok=True)
	checkpoints = {}

	for step in range(num_eras):

		# Prepare Data for Training
		batch = mixed_batching(config, train, groups_train)
		trans, labels = batch_to_tensor(config, batch, doc_key="Description")
		feed_dict = {
			get_tensor(graph, "x:0") : trans,
			get_tensor(graph, "y:0") : labels
		}

		# Run Training Step
		sess.run(get_op(graph, "trainer/optimizer"), feed_dict=feed_dict)
		sess.run(get_op(graph, "bn_applier"), feed_dict=feed_dict)

		# Log Batch Accuracy for Tracking
		if step % 1000 == 0:
			predictions = sess.run(get_tensor(graph, "model:0"), feed_dict=feed_dict)
			logging.info("Minibatch accuracy: %.1f%%" % accuracy(predictions, labels))

		# Log Progress and Save
		if step != 0 and step % epochs == 0:

			learning_rate = get_variable(graph, "lr:0")
			logging.info("Testing for era %d" % (step / epochs))
			logging.info("Learning rate at epoch %d: %g" % (step + 1, sess.run(learning_rate)))

			# Evaluate Model and Visualize
			model = get_tensor(graph, "model:0")
			test_accuracy = evaluate_testset(config, graph, sess, model, test, doc_key="Description")

			# Save Checkpoint
			current_era = int(step / epochs)
			meta_path = save_dir + "era_" + str(current_era) + ".ckpt.meta"
			model_path = saver.save(sess, save_dir + "era_" + str(current_era) + ".ckpt")
			logging.info("Checkpoint saved in file: %s" % model_path)
			checkpoints[current_era] = model_path

			# Stop Training if Converged
			if test_accuracy > best_accuracy:
				best_era = current_era
				best_accuracy = test_accuracy

			if current_era - best_era == 3:
				model_path = checkpoints[best_era]
				break

		# Log Loss and Update TensorBoard
		if step % logging_interval == 0:
			loss = sess.run(get_tensor(graph, "trainer/loss:0"), feed_dict=feed_dict)
			logging.info("Train loss at epoch {0:>8}: {1:3.7f}".format(step + 1, loss))

		# Update Learning Rate
		if step != 0 and step % learning_rate_interval == 0:
			learning_rate = get_variable(graph, "lr:0")
			sess.run(learning_rate.assign(learning_rate / 2))

	# Clean Up Directory
	dataset_path = os.path.basename(dataset).split(".")[0]
	final_model_path = base + dataset_path + ".ckpt"
	final_meta_path = base + dataset_path + ".meta"
	logging.info("Moving final model from {0} to {1}.".format(model_path, final_model_path))
	os.rename(model_path, final_model_path)
	os.rename(meta_path, final_meta_path)
	logging.info("Deleting unneeded directory of checkpoints at {0}".format(save_dir))
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
			_, test, _ = load_data(config)
			evaluate_testset(config, graph, sess, model, test, doc_key="Description")

def run_from_command_line():
	"""Run module from command line"""
	logging.basicConfig(level=logging.INFO)
	config = validate_config(sys.argv[1])
	graph, saver = build_graph(config)
	run_session(config, graph, saver)

if __name__ == "__main__":
	run_from_command_line()
