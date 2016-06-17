#!/usr/local/bin/python3
# pylint: disable=too-many-locals

"""Train a CNN using tensorFlow

Created on Apr 16, 2016
@author: Matthew Sevrens
@author: Tina Wu
@author: J. Andrew Key
@author: Oscar Pan
"""

############################################# USAGE ###############################################

# meerkat.classification.ensemble_cnns [config_file]
# meerkat.classification.ensemble_cnns meerkat/classification/config/ensemble_cnns_config.json
# meerkat.classification.ensemble_cnns meerkat/classification/config/distillation_config.json

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
import shutil
import sys

import numpy as np
import tensorflow as tf

from .tools import (batch_normalization, chunks, max_pool, get_cost_list, string_to_tensor,
	accuracy, get_tensor, get_op, get_variable, threshold, bias_variable, weight_variable, conv2d)
from meerkat.classification.tensorflow_cnn import (run_session, mixed_batching, load_data,
	validate_config)
from meerkat.various_tools import load_params, validate_configuration

logging.basicConfig(level=logging.INFO)

def softmax_with_temperature(tensor, temperature):
	"""return a probability array derived by its logit/temperature"""
	return tf.div(tf.exp(tensor/temperature),
				tf.reduce_sum(tf.exp(tensor/temperature)), name="softmax_flat")

def ensemble_evaluate_testset(config, graph, sess, model, test):
	"""Check error on test set"""

	num_cnns = config["num_cnns"]
	total_count = len(test.index)
	correct_count = 0
	individual_correct_count = [0 for i in range(num_cnns)]
	chunked_test = chunks(np.array(test.index), 128)
	num_chunks = len(chunked_test)

	for i in range(num_chunks):

		batch_test = test.loc[chunked_test[i]]

		trans_test, labels_test, _ = batch_to_tensor(config, batch_test)
		feed_dict_test = {get_tensor(graph, "x:0"): trans_test}
		output = [sess.run(model[j], feed_dict=feed_dict_test) for j in range(num_cnns)]

		individual_batch_correct_count = [np.sum(np.argmax(output[j], 1) == np.argmax(labels_test, 1))
			for j in range(num_cnns)]
		individual_correct_count = [sum(x) for x in
			zip(individual_correct_count, individual_batch_correct_count)]

		ensemble_output = sum(output) / (num_cnns + 0.0)
		batch_correct_count = np.sum(np.argmax(ensemble_output, 1) == np.argmax(labels_test, 1))
		correct_count += batch_correct_count

	for i in range(num_cnns):
		test_accuracy = 100 * (individual_correct_count[i] / (total_count + 0.0))
		logging.info("Test accuracy of model" + str(i+1) + ": %.2f%%" % test_accuracy)
		logging.info("Correct count: " + str(individual_correct_count[i]))
		logging.info("Total count: " + str(total_count))
	test_accuracy = 100.0 * (correct_count / total_count)
	logging.info("Average Ensemble test accuracy: %.2f%%" % test_accuracy)
	logging.info("Correct count: " + str(correct_count))
	logging.info("Total count: " + str(total_count))

	return test_accuracy

def logsoftmax(softmax, name):
	"""Return log of the softmax"""
	return tf.log(tf.clip_by_value(softmax, 1e-10, 1.0), name=name)

def build_graph(config):
	"""Build CNN"""

	doc_length = config["doc_length"]
	alphabet_length = config["alphabet_length"]
	reshape = config["reshape"]
	num_labels = config["num_labels"]
	base_rate = config["base_rate"]
	soft_target = config["soft_target"]
	graph = tf.Graph()
	num_cnns = config["num_cnns"]

	# Get Cost Weights
	cost_list = get_cost_list(config)
	for i, cost in enumerate(cost_list):
		logging.info("Cost for class {0} is {1}".format(i+1, cost))

	# Create Graph
	with graph.as_default():

		learning_rate = tf.Variable(base_rate, trainable=False, name="lr")
		test_accuracy = tf.Variable(0, trainable=False, name="test_accuracy")
		tf.scalar_summary('test_accuracy', test_accuracy)

		input_shape = [None, 1, doc_length, alphabet_length]
		output_shape = [None, num_labels]

		trans_placeholder = tf.placeholder(tf.float32, shape=input_shape, name="x")
		labels_placeholder = tf.placeholder(tf.float32, shape=output_shape, name="y")
		soft_labels_placeholder = tf.placeholder(tf.float32, shape=output_shape, name="y_soft")

		# Utility for Batch Normalization
		layer_sizes = [256] * 8 + [1024, num_labels]
		ewma = tf.train.ExponentialMovingAverage(decay=0.99)
		bn_assigns = []


		def update_batch_normalization(batch, layer_num, model_num):
			"batch normalize + update average mean and variance of layer l"
			axes = [0] if len(batch.get_shape()) == 2 else [0, 1, 2]
			mean, var = tf.nn.moments(batch, axes=axes)
			assign_mean = running_mean[model_num-1][layer_num-1].assign(mean)
			assign_var = running_var[model_num-1][layer_num-1].assign(var)
			bn_assigns.append(ewma.apply([running_mean[model_num-1][layer_num-1],
				running_var[model_num-1][layer_num-1]]))
			with tf.control_dependencies([assign_mean, assign_var]):
				return (batch - mean) / tf.sqrt(var + 1e-10)

		def layer(*args, **kwargs):
			"""Apply all necessary steps in a ladder layer"""
			input_h, details, layer_name, train, model_num = args[:]
			weights = kwargs.get('weights', None)
			biases = kwargs.get('biases', None)

			# Scope for Visualization with TensorBoard
			with tf.name_scope(layer_name):

				# Preactivatin
				if "conv" in layer_name:
					z_pre = conv2d(input_h, weights)
				elif "pool" in layer_name:
					z_pre = max_pool(input_h)
				elif "fc" in layer_name:
					z_pre = tf.matmul(input_h, weights)

				# if layer_name != "h_fc2":
				details["layer_count"] += 1
				layer_n = details["layer_count"]

				if train:
					normalized_layer = update_batch_normalization(z_pre, layer_n, model_num)
				else:
					mean = ewma.average(running_mean[model_num-1][layer_n-1])
					var = ewma.average(running_var[model_num-1][layer_n-1])
					normalized_layer = batch_normalization(z_pre, mean=mean, var=var)

				# Apply Activation
				if "conv" in layer_name or "fc" in layer_name:
					layer = threshold(normalized_layer + biases)
				else:
					layer = normalized_layer

				return layer

		with tf.name_scope("running_mean"):
			running_mean = [[tf.Variable(tf.zeros([l]), trainable=False) for l in layer_sizes]
				for i in range(num_cnns)]

		with tf.name_scope("running_var"):
			running_var = [[tf.Variable(tf.ones([l]), trainable=False) for l in layer_sizes]
				for i in range(num_cnns)]

		def encoder(inputs, name, model_num, train=False, soft_target=False):
			# Encoder Weights and Biases
			"""Add model layers to the graph"""

			details = {"layer_count": 0}


			h_conv1 = layer(inputs, details, "h_conv1", train, model_num, weights=w_conv1, biases=b_conv1)
			h_pool1 = layer(h_conv1, details, "h_pool1", train, model_num)

			h_conv2 = layer(h_pool1, details, "h_conv2", train, model_num, weights=w_conv2, biases=b_conv2)
			h_pool2 = layer(h_conv2, details, "h_pool2", train, model_num)

			h_conv3 = layer(h_pool2, details, "h_conv3", train, model_num, weights=w_conv3, biases=b_conv3)

			h_conv4 = layer(h_conv3, details, "h_conv4", train, model_num, weights=w_conv4, biases=b_conv4)

			h_conv5 = layer(h_conv4, details, "h_conv5", train, model_num, weights=w_conv5, biases=b_conv5)
			h_pool5 = layer(h_conv5, details, "h_pool5", train, model_num)

			h_reshape = tf.reshape(h_pool5, [-1, reshape])

			h_fc1 = layer(h_reshape, details, "h_fc1", train, model_num, weights=w_fc1, biases=b_fc1)

			if train and not soft_target:
				h_fc1 = tf.nn.dropout(h_fc1, 0.5)

			h_fc2 = layer(h_fc1, details, "h_fc2", train, model_num, weights=w_fc2, biases=b_fc2)

			softmax = tf.nn.softmax(bn_scaler * h_fc2, name=name)
			if not soft_target:
				return softmax
			else:
				temperature = config["temperature"]
				softmax_flat = softmax_with_temperature(bn_scaler * h_fc2, temperature)
				return softmax, softmax_flat


		softmax = []
		network = []
		cnn = []

		for i in range(1, num_cnns+1):
			scope_name = "model" + str(i) * (num_cnns > 1)
			with tf.variable_scope(scope_name):
				bn_scaler = tf.Variable(1.0 * tf.ones([num_labels]))

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

				if not soft_target:
					prob_train = encoder(trans_placeholder, "softmax", i, train=True)
				else:
					prob_train, softmax_flat = encoder(trans_placeholder, "softmax", i, train=True,
						soft_target=soft_target)
				softmax.append(prob_train)
				network.append(logsoftmax(softmax[i-1], "network"))
				prob_full = encoder(trans_placeholder, "softmax_full", i, train=False)
				cnn.append(logsoftmax(prob_full, "cnn"*(num_cnns>1)))

		ensemble = sum(softmax) / (num_cnns + 0.0)
		weighted_labels = cost_list * labels_placeholder

		def cal_loss(sub_softmax, sub_network, name):
			"""Returns negative correlation learning loss function"""
			return tf.neg(tf.reduce_mean(tf.reduce_sum(sub_network * weighted_labels, 1)) +
				0.5 * tf.reduce_mean(tf.reduce_sum((ensemble - sub_softmax) ** 2, 1)), name=name)

		def make_optimizer(loss, op_name, scope_name):
			"""Returns a  momentumOptimizer"""
			return tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, name=op_name,
						var_list=[x for x in tf.trainable_variables() if x.name.startswith(scope_name)])

		with tf.name_scope("trainer"):
			if not soft_target:
				loss = [cal_loss(softmax[i], network[i], "loss"+str(i+1)) for i in range(num_cnns)]
			else:
				loss = [tf.neg(0.85 * tf.reduce_mean(tf.reduce_sum(logsoftmax(softmax_flat,
					"network_flat") * soft_labels_placeholder, 1))
					+ 0.15 * tf.reduce_mean(tf.reduce_sum(network[0] * weighted_labels, 1)),
					name="loss1")]
			optimizer = [make_optimizer(loss[i], "optimizer"+str(i+1), "model"+str(i+1)*(num_cnns>1))
				for i in range(num_cnns)]

		bn_updates = tf.group(*bn_assigns)
		with tf.control_dependencies(optimizer):
			_ = tf.group(bn_updates, name="bn_applier")

		def get_saver(name):
			"""return a saver for namespace name"""
			return tf.train.Saver([x for x in tf.all_variables() if x.name.startswith(name)])

		saver = [get_saver("model"+str(i+1)*(num_cnns>1)) for i in range(num_cnns)]

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
	num_cnns = config["num_cnns"]

	best_accuracy, best_era = 0, 0
	save_dir = "meerkat/classification/models/checkpoints/"
	os.makedirs(save_dir, exist_ok=True)
	checkpoints = {}

	for step in range(num_eras):

		# Prepare Data for Training
		batch = mixed_batching(config, train, groups_train)
		trans, labels, labels_soft = batch_to_tensor(config, batch, soft_target=config["soft_target"])
		feed_dict = {
			get_tensor(graph, "x:0") : trans,
			get_tensor(graph, "y:0") : labels
		}
		if config["soft_target"]:
			feed_dict[get_tensor(graph, "y_soft:0")] = labels_soft

		# Run Training Step
		# sess.run(get_op(graph, "trainer/optimizer"), feed_dict=feed_dict)
		for i in range(1, num_cnns+1):
			sess.run(get_op(graph, "trainer/optimizer"+str(i)), feed_dict=feed_dict)
		sess.run(get_op(graph, "bn_applier"), feed_dict=feed_dict)

		# Log Batch Accuracy for Tracking
		if step % 1000 == 0:
			# Calculate Batch Accuracy
			for i in range(1, num_cnns+1):
				predictions = sess.run(get_tensor(graph, "model"+(str(i)+"/cnn")*(num_cnns>1)+":0"), feed_dict=feed_dict)
				logging.info("Minibatch accuracy for cnn" + str(i) + ": %.1f%%" % accuracy(predictions, labels))
			# Estimate Accuracy for Visualization
			model = [get_tensor(graph, "model"+(str(i)+"/cnn")*(num_cnns>1)+":0") for i in range(num_cnns)]
			ensemble_accuracy = ensemble_evaluate_testset(config, graph, sess, model, test)

		# Log Loss and Update TensorBoard
		if step % logging_interval == 0:
			loss = [sess.run(get_tensor(graph, "trainer/loss"+str(i)+":0"), feed_dict=feed_dict)
				for i in range(1, num_cnns+1)]
			for i in range(num_cnns):
				logging.info("Train loss" + str(i+1) +" at epoch {0:>8}: {1:3.7f}".format(step + 1, loss[i]))

		# Log Progress and Save
		if step != 0 and step % epochs == 0:

			learning_rate = get_variable(graph, "lr:0")
			logging.info("Testing for era %d" % (step / epochs))
			logging.info("Learning rate at epoch %d: %g" % (step + 1, sess.run(learning_rate)))

			# Save Checkpoint
			current_era = int(step / epochs)
			meta_path = [save_dir + "era_" + str(current_era) + "_model"+str(i+1) + ".ckpt.meta"
				for i in range(num_cnns)]
			model_path = [saver[i].save(sess, save_dir +
				"era_" + str(current_era) + "_model"+str(i+1) + ".ckpt") for i in range(num_cnns)]
			checkpoints[current_era] = model_path
			for i in range(num_cnns):
				logging.info("Checkpoint saved in file: %s" % model_path[i])

			# Stop Training if Converged
			if ensemble_accuracy > best_accuracy:
				best_era = current_era
				best_accuracy = ensemble_accuracy

			if current_era - best_era == config["stopping_criterion"]:
				model_path = checkpoints[best_era]
				break


		# Update Learning Rate
		if step != 0 and step % learning_rate_interval == 0:
			learning_rate = get_variable(graph, "lr:0")
			sess.run(learning_rate.assign(learning_rate / 2))

	# Clean Up Directory
	dataset_path = os.path.basename(dataset).split(".")[0]
	save_path = "meerkat/classification/models/ensemble_cnns/"
	os.makedirs(save_path, exist_ok=True)
	for i in range(num_cnns):
		final_model_path = save_path + dataset_path + ".model" + str(i+1) + ".ckpt"
		final_meta_path = save_path + dataset_path + ".model" + str(i+1) + ".meta"
		logging.info("Moving final model from {0} to {1}.".format(model_path[i], final_model_path))

		os.rename(model_path[i], final_model_path)
		os.rename(meta_path[i], final_meta_path)
	logging.info("Deleting unneeded directory of checkpoints at {0}".format(save_dir))
	shutil.rmtree(save_dir)

	if config["soft_target"]:
		return final_model_path

def run_from_command_line():
	"""Run module from command line"""
	logging.basicConfig(level=logging.INFO)
	config = validate_config(sys.argv[1])
	graph, saver = build_graph(config)
	run_session(config, graph, saver)

if __name__ == "__main__":
	run_from_command_line()
