#!/usr/local/bin/python3

import logging
import os
import random
import shutil

import numpy as np
import tensorflow as tf

from .tools import (fill_description_unmasked, reverse_map, batch_normalization, chunks,
	accuracy, get_tensor, get_op, get_variable, threshold, bias_variable, weight_variable, conv2d,
	max_pool, get_cost_list, string_to_tensor)
from meerkat.various_tools import load_params, load_piped_dataframe, validate_configuration
from meerkat.classification.tensorflow_cnn import (mixed_batching, load_data, validate_config, batch_to_tensor)
from meerkat.classification.data_handler import download_data, upload_result

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "52.42.128.45:2222",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "52.38.177.209:2222,52.41.169.166:2222",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

tf.app.flags.DEFINE_string("model_type", "", "One of 'merchant', 'subtype', 'category'")
tf.app.flags.DEFINE_string("bank_or_card", "", "One of 'bank', 'card'")
tf.app.flags.DEFINE_string("credit_or_debit", "", "One of 'credit', 'debit'")

FLAGS = tf.app.flags.FLAGS

logging.basicConfig(level=logging.INFO)

def inference(trans_pl, config):

	doc_length = config["doc_length"]
	alphabet_length = config["alphabet_length"]
	reshape = config["reshape"]
	num_labels = config["num_labels"]

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

	running_mean = [tf.Variable(tf.zeros([l]), trainable=False) for l in layer_sizes]

	running_var = [tf.Variable(tf.ones([l]), trainable=False) for l in layer_sizes]

	def layer(*args, **kwargs):
		"""Apply all necessary steps in a ladder layer"""
		input_h, details, layer_name, train = args[:]
		weights = kwargs.get('weights', None)
		biases = kwargs.get('biases', None)

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

	network = encoder(trans_pl, "network", train=True)
	model = encoder(trans_pl, "model", train=False)

	return network, model, bn_assigns

def placeholder_inputs(config):

				# [batch, height, width, channels]
	input_shape = [None, 1, config["doc_length"], config["alphabet_length"]]
	output_shape = [None, config["num_labels"]]

	trans_placeholder = tf.placeholder(tf.float32, shape=input_shape, name='x')
	labels_placeholder = tf.placeholder(tf.float32, shape=output_shape, name='y')

	return trans_placeholder, labels_placeholder

def evaluate_testset(config, trans_pl, sess, model, test):
	"""Check error on test set"""

	total_count = len(test.index)
	correct_count = 0
	chunked_test = chunks(np.array(test.index), 128)
	num_chunks = len(chunked_test)

	for i in range(num_chunks):

		batch_test = test.loc[chunked_test[i]]
		batch_size = len(batch_test)

		trans_test, labels_test = batch_to_tensor(config, batch_test)
		feed_dict_test = {trans_pl: trans_test}
		output = sess.run(model, feed_dict=feed_dict_test)

		batch_correct_count = np.sum(np.argmax(output, 1) == np.argmax(labels_test, 1))

		correct_count += batch_correct_count

	te_accuracy = 100.0 * (correct_count / total_count)
	logging.info("Test accuracy: %.2f%%" % te_accuracy)
	logging.info("Correct count: " + str(correct_count))
	logging.info("Total count: " + str(total_count))

	return te_accuracy

def get_meta(config, base):

	with tf.Session() as sess:
		cost_list = get_cost_list(config)
		trans_placeholder, labels_placeholder = placeholder_inputs(config)
		network, model, bn_assigns = inference(trans_placeholder, config)
		global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
		learning_rate = tf.train.exponential_decay(config["base_rate"], global_step, 15000, 0.5, staircase=True)

		# Calculate Loss and Optimize
		weighted_labels = cost_list * labels_placeholder
		loss = tf.neg(tf.reduce_mean(tf.reduce_sum(network * weighted_labels, 1)), name="loss")
		optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step, name="optimizer")
		tf.scalar_summary('loss', loss)

		bn_updates = tf.group(*bn_assigns)
		with tf.control_dependencies([optimizer]):
			bn_applier = tf.group(bn_updates, name="bn_applier")

		saver = tf.train.Saver()
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		saver.save(sess, base + 'train.ckpt')
	os.remove(base + 'checkpoint')
	os.remove(base + 'train.ckpt')
	return base + 'train.ckpt.meta'

def train(config, target, cluster, meta_path):

	is_chief = (FLAGS.task_index == 0)

	# Assigns ops to the local worker by default.
	with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):

		cost_list = get_cost_list(config)
		for i, cost in enumerate(cost_list):
			logging.info("Cost for class {0} is {1}".format(i+1, cost))

		trans_placeholder, labels_placeholder = placeholder_inputs(config)
		network, model, bn_assigns = inference(trans_placeholder, config)
		global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
		learning_rate = tf.train.exponential_decay(config["base_rate"], global_step, 15000, 0.5, staircase=True)

		# Calculate Loss and Optimize
		weighted_labels = cost_list * labels_placeholder
		loss = tf.neg(tf.reduce_mean(tf.reduce_sum(network * weighted_labels, 1)), name="loss")
		optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step, name="optimizer")
		tf.scalar_summary('loss', loss)

		bn_updates = tf.group(*bn_assigns)
		with tf.control_dependencies([optimizer]):
			bn_applier = tf.group(bn_updates, name="bn_applier")

		saver = tf.train.Saver()
		summary_op = tf.merge_all_summaries()
		init_op = tf.initialize_all_variables()

		sv = tf.train.Supervisor(is_chief=is_chief, logdir="/tmp/train_logs", init_op=init_op, \
			summary_op=None, saver=saver, global_step=global_step, save_model_secs=600)

		with sv.managed_session(target) as sess:

			epochs = config["epochs"]
			eras = config["eras"]
			dataset = config["dataset"]
			train, test, groups_train = load_data(config)
			num_eras = epochs * eras
			logging_interval = 50

			best_accuracy, best_era = 0, 0
			base = "meerkat/classification/models/"
			save_dir = base + "checkpoints/"
			if is_chief: os.makedirs(save_dir, exist_ok=True)
			checkpoints = {}

			local_step = 0
			while True:

				# Prepare Data for Training
				batch = mixed_batching(config, train, groups_train)
				trans, labels = batch_to_tensor(config, batch)
				feed_dict = {
					trans_placeholder : trans,
					labels_placeholder : labels
				}

				# Run Training Step
				_, step = sess.run([optimizer, global_step], feed_dict=feed_dict)
				sess.run(bn_applier, feed_dict=feed_dict)
				local_step += 1

				# Log Batch Accuracy for Tracking
				if local_step % 1000 == 0:
					predictions = sess.run(model, feed_dict=feed_dict)
					logging.info("Minibatch accuracy: %.1f%%" % accuracy(predictions, labels))

				# Log Progress and Save
				if local_step != 0 and local_step % epochs == 0:

					logging.info("Testing for era %d" % (local_step / epochs))
					logging.info("Learning rate at epoch %d: %g" % (local_step, sess.run(learning_rate)))

					# Evaluate Model and Visualize
					te_accuracy = evaluate_testset(config, trans_placeholder, sess, model, test)

					# Save Checkpoint
					current_era = int(local_step / epochs)
					if is_chief:
#						tmp_meta_path = "/tmp/train_logs/" + "era_" + str(current_era) + ".ckpt.meta"
						tmp_model_path = saver.save(sess, "/tmp/train_logs/" + "era_" + str(current_era) + ".ckpt")
#						meta_path = save_dir + "era_" + str(current_era) + ".ckpt.meta"
						model_path = save_dir + "era_" + str(current_era) + ".ckpt"
#						os.rename(tmp_meta_path, meta_path)
						os.rename(tmp_model_path, model_path)
						logging.info("Checkpoint saved in file: %s" % model_path)
						checkpoints[current_era] = model_path

					# Stop Training if Converged
					if te_accuracy > best_accuracy:
						best_era = current_era
						best_accuracy = te_accuracy

					if current_era - best_era == 3:
						if is_chief:
							model_path = checkpoints[best_era]
						break

				# Log Loss and Update TensorBoard
				if local_step % logging_interval == 0:
					loss_ = sess.run(loss, feed_dict=feed_dict)
					logging.info("Train loss at epoch {0:>8}: {1:3.7f}".format(local_step, loss_))

				if step > num_eras: break

			if is_chief:
				# Clean Up Directory
				dataset_path = os.path.basename(dataset).split(".")[0]
				final_model_path = base + dataset_path + ".ckpt"
				final_meta_path = base + dataset_path + ".meta"
				logging.info("Moving final model from {0} to {1}.".format(model_path, final_model_path))
				os.rename(model_path, final_model_path)
				os.rename(meta_path, final_meta_path)
				logging.info("Deleting unneeded directory of checkpoints at {0}".format(save_dir))
				shutil.rmtree(save_dir)

		sv.stop()

		return final_model_path if is_chief else None

def main(_):
	logging.basicConfig(level=logging.INFO)

	ps_hosts = FLAGS.ps_hosts.split(",")
	worker_hosts = FLAGS.worker_hosts.split(",")

	# Create a cluster from the parameter server and worker hosts.
	cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

	# Create and start a server for the local task.
	server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

	if FLAGS.job_name == "ps":
		server.join()
	elif FLAGS.job_name == "worker":

		config, test_file, s3_params = download_data(FLAGS.model_type, FLAGS.bank_or_card, FLAGS.credit_or_debit)
		meta_path = "meerkat/classification/models/train.ckpt.meta"
		final_model_path = train(config, server.target, cluster, meta_path)
		if FLAGS.task_index == 0:
			upload_result(config, final_model_path, test_file, s3_params)
		else:
			shutil.rmtree(s3_params["save_path"])


if __name__ == "__main__":
	if tf.gfile.Exists("/tmp/train_logs"):
		tf.gfile.DeleteRecursively("/tmp/train_logs")
	tf.app.run()
