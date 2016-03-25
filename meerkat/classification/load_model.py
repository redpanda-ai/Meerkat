#!/usr/local/bin/python3.3

"""This module loads classifier from various libraries and produces 
helper functions that will classify transactions. Depending on the model 
requested this module will load a different previously generated model.

Created on Feb 25, 2014
@author: Matthew Sevrens
"""

import sys
import logging

import numpy as np
import tensorflow as tf
from sklearn.externals import joblib

from meerkat.classification.tensorflow_cnn import build_graph, validate_config, get_tensor, string_to_tensor

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

def load_tensorflow_model(model_name):
	"""Load a tensorFlow module by name"""

	# Switch on Models
	if model_name == "card_debit_subtype":
		model_path = "meerkat/classification/models/card_debit_subtype.ckpt"
		config_path = "config/tf_cnn_config.json"
	else:
		logging.warning("Model not found. Terminating")
		sys.exit()

	# Load Graph
	config = validate_config(config_path)
	graph, saver = build_graph(config)
	label_map = config["label_map"]

	# Load Session and Graph
	sess = tf.Session(graph=graph)
	saver.restore(sess, model_path)
	model = get_tensor(graph, "model:0")
	
	# Generate Helper Function
	def apply_cnn(trans, doc_key="description", label_key="CNN", label_only=True):
		"""Apply CNN to transactions"""

		alphabet_length = config["alphabet_length"]
		doc_length = config["doc_length"]
		batch_size = len(trans)

		tensor = np.zeros(shape=(batch_size, 1, alphabet_length, doc_length))

		for index, doc in enumerate(trans):
			tensor[index][0] = string_to_tensor(config, doc[doc_key], doc_length)

		tensor = np.transpose(tensor, (0, 1, 3, 2))
		feed_dict_test = {get_tensor(graph, "x:0"): tensor}
		output = sess.run(model, feed_dict=feed_dict_test)
		labels = np.argmax(output, 1)
	
		for index, transaction in enumerate(trans):
			label = label_map.get(str(labels[index]), "")
			if isinstance(label, dict) and label_only: label = label["label"]
			transaction[label_key] = label

		return trans

	return apply_cnn

if __name__ == "__main__":
	# pylint:disable=pointless-string-statement
	"""Print a warning to not execute this file as a module"""
	logging.warning("This module is a library that contains useful functions;" +\
	 "it should not be run from the console.")
