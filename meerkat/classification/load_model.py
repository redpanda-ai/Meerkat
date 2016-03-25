#!/usr/local/bin/python3.3

"""This module loads classifier from various libraries and produces 
helper functions that will classify transactions. Depending on the model 
requested this module will load a different previously generated model.

Created on Feb 25, 2014
@author: Matthew Sevrens
"""

import sys
import logging

import tensorflow as tf
from sklearn.externals import joblib

from meerkat.classification.tensorflow_cnn import build_graph

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
		model_path = "card_debit_subtype.ckpt"
	else:
		logging.warning("Model not found. Terminating")
		sys.exit()

	# Load Graph
	graph, saver = build_graph()

	# Load Session and Graph
	tf.initialize_all_variables().run()
	sess = tf.Session(graph=graph)
	saver.restore(sess, model_path)
	model = get_tensor(graph, "model:0")

	return model

if __name__ == "__main__":
	# pylint:disable=pointless-string-statement
	"""Print a warning to not execute this file as a module"""
	logging.warning("This module is a library that contains useful functions;" +\
	 "it should not be run from the console.")
