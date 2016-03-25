#!/usr/local/bin/python3.3

"""This module loads classifier from various libraries and produces 
helper functions that will classify transactions. Depending on the model 
requested this module will load a different previously generated model.

Created on Feb 25, 2014
@author: Matthew Sevrens
"""

from sklearn.externals import joblib
import logging
import sys

def select_model(mode):
	"""Load either Card or Bank classifier depending on
	requested model"""

	# Switch on Models
	if mode == "card_sws":
		logging.warning("--- Loading Card SWS Model ---")
		model_path = "meerkat/classification/models/final_card_sws.pkl"
	elif mode == "bank_sws":
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

if __name__ == "__main__":
	# pylint:disable=pointless-string-statement
	"""Print a warning to not execute this file as a module"""
	logging.warning("This module is a library that contains useful functions;" +\
	 "it should not be run from the console.")
