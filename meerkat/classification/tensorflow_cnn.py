#!/usr/local/bin/python3
# pylint: disable=unused-variable
# pylint: disable=too-many-locals

"""Train a CNN using tensorFlow

Created on Apr 16, 2016
@author: Matthew Sevrens
@author: Tina Wu
@author: J. Andrew Key
"""

############################################# USAGE ###############################################

# meerkat.classification.tensorflow_cnn [config_file]
# meerkat.classification.tensorflow_cnn meerkat/classification/config/default_tf_config.json

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

from meerkat.classification.tools import fill_description_unmasked, reverse_map
from meerkat.various_tools import load_params, load_piped_dataframe, validate_configuration

logging.basicConfig(level=logging.INFO)

def chunks(array, num):
	"""Chunk array into equal sized parts"""
	num = max(1, num)
	return [array[i:i + num] for i in range(0, len(array), num)]

def validate_config(config):
	"""Validate input configuration"""

	config = load_params(config)
	schema_file = "meerkat/classification/config/tensorflow_cnn_schema.json"
	config = validate_configuration(config, schema_file)
	logging.debug("Configuration is :\n{0}".format(pprint.pformat(config)))
	# about this reshape size, read this wiki page below.
	# https://github.com/joeandrewkey/Meerkat/wiki/Using-Correct-Tensor-Size
	reshape = ((config["doc_length"] - 96) / 27) * 256
	config["reshape"] = int(reshape)
	config["label_map"] = load_params(config["label_map"])
	config["num_labels"] = len(config["label_map"].keys())
	config["alpha_dict"] = {a : i for i, a in enumerate(config["alphabet"])}
	config["base_rate"] = config["base_rate"] * math.sqrt(config["batch_size"]) / math.sqrt(128)
	config["alphabet_length"] = len(config["alphabet"])

	return config

def load_data(config):
	"""Load labeled data and label map"""

	model_type = config["model_type"]
	dataset = config["dataset"]
	label_map = config["label_map"]
	ledger_entry = config["ledger_entry"]

	ground_truth_labels = {
		'category' : 'PROPOSED_CATEGORY',
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

	if model_type != "merchant":
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

	return train, test, groups_train

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
	batch_size = len(batch.index)

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

def evaluate_testset(config, graph, sess, model, test):
	"""Check error on test set"""

	with tf.name_scope('accuracy'):

		total_count = len(test.index)
		correct_count = 0
		chunked_test = chunks(np.array(test.index), 128)
		num_chunks = len(chunked_test)

		for i in range(num_chunks):

			batch_test = test.loc[chunked_test[i]]
			batch_size = len(batch_test)

			trans_test, labels_test = batch_to_tensor(config, batch_test)
			feed_dict_test = {get_tensor(graph, "x:0"): trans_test}
			output = sess.run(model, feed_dict=feed_dict_test)

			batch_correct_count = np.sum(np.argmax(output, 1) == np.argmax(labels_test, 1))

			correct_count += batch_correct_count
		
		test_accuracy = 100.0 * (correct_count / total_count)
		logging.info("Test accuracy: %.2f%%" % test_accuracy)
		logging.info("Correct count: " + str(correct_count))
		logging.info("Total count: " + str(total_count))

		accuracy = get_variable(graph, "test_accuracy:0")
		sess.run(accuracy.assign(test_accuracy))

	return test_accuracy
	
def accuracy(predictions, labels):
	"""Return accuracy for a batch"""
	return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

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

def threshold(tensor):
	"""ReLU with threshold at 1e-6"""
	return tf.mul(tf.to_float(tf.greater_equal(tensor, 1e-6)), tensor)

def bias_variable(shape, flat_input_shape):
	"""Initialize biases"""
	stdv = 1 / math.sqrt(flat_input_shape)
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

def get_cost_list(config):
	"""Retrieve a cost matrix"""

	# Get the class numbers sorted numerically
	label_map = config["label_map"]
	keys = sorted([int(x) for x in label_map.keys()])

	# Produce an ordered list of cost values
	cost_list = []
	for key in keys:
		cost = label_map[str(key)].get("cost", 1.0)
		cost_list.append(cost)

	return cost_list

def build_graph(config):
	"""Build CNN"""

	doc_length = config["doc_length"]
	alphabet_length = config["alphabet_length"]
	reshape = config["reshape"]
	num_labels = config["num_labels"]
	base_rate = config["base_rate"]
	batch_size = config["batch_size"]
	graph = tf.Graph()

	# Get Cost Weights
	cost_list = get_cost_list(config)
	for i, cost in enumerate(cost_list):
		logging.info("Cost for class {0} is {1}".format(i+1, cost))

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

		def layer(input_h, details, layer_name, train, weights=None, biases=None):
			"""Apply all necessary steps in a ladder layer"""

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
					z = update_batch_normalization(z_pre, layer_n)
				else:
					mean = ewma.average(running_mean[layer_n-1])
					var = ewma.average(running_var[layer_n-1])
					z = batch_normalization(z_pre, mean=mean, var=var)

				# Apply Activation
				if "conv" in layer_name or "fc" in layer_name:
					layer = threshold(z + biases)
				else:
					layer = z

			return layer

		def batch_normalization(batch, mean=None, var=None):
			"""Perform batch normalization"""
			if mean is None or var is None:
				axes = [0] if len(batch.get_shape()) == 2 else [0, 1, 2]
				mean, var = tf.nn.moments(batch, axes=axes)
			return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

		def update_batch_normalization(batch, l):
			"batch normalize + update average mean and variance of layer l"
			axes = [0] if len(batch.get_shape()) == 2 else [0, 1, 2]
			mean, var = tf.nn.moments(batch, axes=axes)
			assign_mean = running_mean[l-1].assign(mean)
			assign_var = running_var[l-1].assign(var)
			bn_assigns.append(ewma.apply([running_mean[l-1], running_var[l-1]]))
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
			weighted_labels = cost_list * labels_placeholder
			loss = tf.neg(tf.reduce_mean(tf.reduce_sum(network * weighted_labels, 1)), name="loss")
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
	save_dir = "meerkat/classification/models/checkpoints/"
	os.makedirs(save_dir, exist_ok=True)
	checkpoints = {}

	# Visualize Using TensorBoard
	merged = tf.merge_all_summaries()
	writer = tf.train.SummaryWriter("/home/ubuntu/tensorboard_log", graph)

	for step in range(num_eras):

		# Prepare Data for Training
		batch = mixed_batching(config, train, groups_train)
		trans, labels = batch_to_tensor(config, batch)
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
			test_accuracy = evaluate_testset(config, graph, sess, model, test)

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
			summary = sess.run(merged, feed_dict=feed_dict)
			writer.add_summary(summary, step)

		# Update Learning Rate
		if step != 0 and step % learning_rate_interval == 0:
			learning_rate = get_variable(graph, "lr:0")
			sess.run(learning_rate.assign(learning_rate / 2))

	# Clean Up Directory
	dataset_path = os.path.basename(dataset).split(".")[0]
	final_model_path = "meerkat/classification/models/" + dataset_path + ".ckpt"
	final_meta_path = "meerkat/classification/models/" + dataset_path + ".meta"
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
			evaluate_testset(config, graph, sess, model, test)

def run_from_command_line():
	"""Run module from command line"""
	logging.basicConfig(level=logging.INFO)
	config = validate_config(sys.argv[1])
	graph, saver = build_graph(config)
	run_session(config, graph, saver)

if __name__ == "__main__":
	run_from_command_line()
