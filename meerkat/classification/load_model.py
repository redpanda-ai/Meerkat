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

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.externals import joblib
from meerkat.various_tools import load_params
from meerkat.classification.auto_load import main_program as load_models_from_s3
from meerkat.classification.tensorflow_cnn import (validate_config, get_tensor, string_to_tensor)

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

	# Switch on Models
	if model_name == "bank_merchant":
		model_path = base + "merchant.bank.ckpt"
		label_map_path = label_map_base + "merchant.bank.json"
	elif model_name == "card_merchant":
		model_path = base + "merchant.card.ckpt"
		label_map_path = label_map_base + "merchant.card.json"
	elif model_name == "bank_debit_subtype":
		model_path = base + "subtype.bank.debit.ckpt"
		label_map_path = label_map_base + "subtype.bank.debit.json"
	elif model_name == "bank_credit_subtype":
		model_path = base + "subtype.bank.credit.ckpt"
		label_map_path = label_map_base + "subtype.bank.credit.json"
	elif model_name == "card_debit_subtype":
		model_path = base + "subtype.card.debit.ckpt"
		label_map_path = label_map_base + "subtype.card.debit.json"
	elif model_name == "card_credit_subtype":
		model_path = base + "subtype.card.credit.ckpt"
		label_map_path = label_map_base + "subtype.card.credit.json"
	else:
		logging.warning("Model not found. Terminating")
		sys.exit()

	return get_tf_cnn_by_path(model_path, label_map_path, gpu_mem_fraction=gpu_mem_fraction)

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

		alphabet_length = config["alphabet_length"]
		if "merchant" not in config["model_path"]:
			doc_length = config["doc_length"]
		else:
			doc_length = 123
		batch_size = len(trans)

		tensor = np.zeros(shape=(batch_size, 1, alphabet_length, doc_length))

		for index, doc in enumerate(trans):
			tensor[index][0] = string_to_tensor(config, doc[doc_key], doc_length)

		tensor = np.transpose(tensor, (0, 1, 3, 2))
		feed_dict_test = {get_tensor(graph, "x:0"): tensor}
		output = sess.run(model, feed_dict=feed_dict_test)
		if not soft_target:
			labels = np.argmax(output, 1) + 1

			for index, transaction in enumerate(trans):
				label = label_map.get(str(labels[index]), "")
				if isinstance(label, dict) and label_only:
					label = label["label"]
				transaction[label_key] = label
		else:
			return output

		return trans

	return apply_cnn

if __name__ == "__main__":
	# pylint:disable=pointless-string-statement
	"""Print a warning to not execute this file as a module"""
	logging.warning("This module is a library that contains useful functions;" +\
	 "it should not be run from the console.")
