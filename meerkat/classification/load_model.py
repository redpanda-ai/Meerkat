#!/usr/local/bin/python3.3

"""This module loads classifier from various libraries and produces 
helper functions that will classify transactions. Depending on the model 
requested this module will load a different previously generated model.

Created on Feb 25, 2014
@author: Matthew Sevrens
"""

from os.path import isfile
import sys
import logging
import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops
from sklearn.externals import joblib
from meerkat.various_tools import load_params
from meerkat.classification.auto_load import main_program as load_models_from_s3
from meerkat.classification.tensorflow_cnn import (validate_config, get_tensor, string_to_tensor)
from meerkat.longtail.bilstm_tagger import trans_to_tensor, get_tags
from meerkat.longtail.bilstm_tagger import validate_config as bilstm_validate_config

def load_scikit_model(model_name):
	"""Load either Card or Bank classifier depending on
	requested model"""

	# Switch on Models
	if model_name == "card_sws":
		logging.warning("--- Loading Card SWS Model ---")
		model_path = "meerkat/classification/models/final_card_sws.pkl"
	elif model_name == "bank_sws":
		logging.warning("--- Loading Bank SWS Model ---")
		model_path = "meerkat/classification/models/final_bank_sws.pkl"
	else:
		logging.warning("--- Classifier requested not found. Terminating ---")
		sys.exit()

	# Load Model
	model = joblib.load(model_path)

	# Generate Helper Function
	def classifier(description):
		"""classify the variable description with Card or Bank model"""
		result = list(model.predict([description]))[0]
		return result
			
	return classifier

def get_tf_cnn_by_name(model_name, gpu_mem_fraction=False):
	"""Load a tensorFlow CNN by name"""

	base = "meerkat/classification/models/"
	label_map_base = "meerkat/classification/label_maps/"

	model_names = ["bank_merchant", "card_merchant", "bank_debit_subtype", "bank_credit_subtype",
		"card_debit_subtype", "card_credit_subtype", "bank_debit_category", "bank_credit_category",
		"card_debit_category", "card_credit_category"]
	if model_name in model_names:
		temp = model_name.split("_")
		model_type = temp[-1] + '.' + '.'.join(temp[:-1])
		model_path = base + model_type + ".ckpt"
		label_map_path = label_map_base + model_type + ".json"
	else:
		logging.warning("Model not found. Terminating")
		sys.exit()

	return get_tf_cnn_by_path(model_path, label_map_path, gpu_mem_fraction=gpu_mem_fraction)

def get_tf_rnn_by_path(model_path, w2i_path, gpu_mem_fraction=False, model_name=False):
	"""Load a tensorflow rnn model"""

	config_path = "meerkat/longtail/bilstm_config.json"
	if not isfile(model_path):
		logging.warning("Resources to load model not found.")
		sys.exit()

	# Load Graph
	config = bilstm_validate_config(config_path)
	config["model_path"] = model_path
	meta_path = model_path.split(".ckpt")[0] + ".meta"
	config["w2i"] = load_params(w2i_path)

	# Load Session and Graph
	ops.reset_default_graph()
	saver = tf.train.import_meta_graph(meta_path)

	if gpu_mem_fraction:
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
	else:
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, device_count={"GPU":0}))

	saver.restore(sess, config["model_path"])
	graph = sess.graph

	if not model_name:
		model = get_tensor(graph, "model:0")
	else:
		model = get_tensor(graph, model_name)

	# Generate Helper Function
	def apply_rnn(trans, doc_key="Description", label_key="Predicted", name_only=True, tags=False):
		"""Apply RNN to transactions"""

		for _, doc in enumerate(trans):
			if tags:
				# if tags, tag all tokens with get_tags for evaluation purposes
				tran, label = get_tags(config, doc)
				doc["ground_truth"] = label
			else:
				tran = doc[doc_key].lower().split()[0:config["max_tokens"]]
			char_inputs, word_lengths, word_indices, _ = trans_to_tensor(config, tran)
			feed_dict = {
				get_tensor(graph, "char_inputs:0"): char_inputs,
				get_tensor(graph, "word_inputs:0"): word_indices,
				get_tensor(graph, "word_lengths:0"): word_lengths,
				get_tensor(graph, "trans_length:0"): len(tran),
				get_tensor(graph, "train:0"): False
			}

			output = sess.run(model, feed_dict=feed_dict)
			if name_only:
				# return merchant name if name_only else return tags of all tokens
				output = [config["tag_map"][str(i)] for i in np.argmax(output, 1)]
				target_indices = [i for i in range(len(output)) if output[i] == "merchant"]
				doc[label_key] = " ".join([tran[i] for i in target_indices])
			else:
				doc[label_key] = output

		return trans

	return apply_rnn

def get_tf_cnn_by_path(model_path, label_map_path, gpu_mem_fraction=False, model_name=False):
	"""Load a tensorFlow module by name"""

	# Load Config
	config_path = "meerkat/classification/config/default_tf_config.json"

	# Validate Model and Label Map
	if not isfile(model_path):
		logging.warning("Resouces to load model not found. Loading from S3")
		load_models_from_s3()

	# Load Graph
	config = load_params(config_path)
	config["label_map"] = label_map_path
	config["model_path"] = model_path
	meta_path = model_path.split(".ckpt")[0] + ".meta"
	config = validate_config(config)
	label_map = config["label_map"]

	# Load Session and Graph
	ops.reset_default_graph()
	saver = tf.train.import_meta_graph(meta_path)

	if gpu_mem_fraction:
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
	else:
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

	saver.restore(sess, config["model_path"])
	graph = sess.graph

	if not model_name:
		model = get_tensor(graph, "model:0")
	else:
		model = get_tensor(graph, model_name)

	# Generate Helper Function
	def apply_cnn(trans, doc_key="description", label_key="CNN", label_only=True, soft_target=False):
		"""Apply CNN to transactions"""

		doc_length = config["doc_length"]
		batch_size = len(trans)

		tensor = np.zeros(shape=(batch_size, 1, config["alphabet_length"], doc_length))

		for index, doc in enumerate(trans):
			tensor[index][0] = string_to_tensor(config, doc[doc_key], doc_length)

		tensor = np.transpose(tensor, (0, 1, 3, 2))
		feed_dict_test = {get_tensor(graph, "x:0"): tensor}
		output = sess.run(model, feed_dict=feed_dict_test)
		if not soft_target:
			labels = np.argmax(output, 1) + 1
			scores = np.amax(output, 1)

			for index, transaction in enumerate(trans):
				label = label_map.get(str(labels[index]), "")
				if ('threshold' in label and label["threshold"] and
					scores[index] < math.log(float(label["threshold"]))):
					label = label_map.get('1', '') # first class in label map should always be the default one
				if isinstance(label, dict) and label_only:
					label = label["label"]
				transaction[label_key] = label

				# Append score for transaction under debug mode
				if label_key == "CNN":
					transaction["merchant_score"] = "{0:.6}".format(math.exp(scores[index]))
				if label_key == "subtype_CNN":
					transaction["subtype_score"] = "{0:.6}".format(math.exp(scores[index]))
				if label_key == "category_CNN":
					transaction["category_score"] = "{0:.6}".format(math.exp(scores[index]))

		else:
			return output

		return trans

	return apply_cnn

if __name__ == "__main__":
	# pylint:disable=pointless-string-statement
	"""Print a warning to not execute this file as a module"""
	logging.warning("This module is a library that contains useful functions;" +\
	 "it should not be run from the console.")
