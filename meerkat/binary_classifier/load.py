#!/usr/local/bin/python3
"""This module loads our general classifier."""

from sklearn.externals import joblib

def predict_if_physical_transaction(description=None):

	"""This method uses a previously generated model to
	classify a single transaction as either physical 
	or non physical"""

	result = list(GRID_SEARCH.predict([description]))[0]

	return result

# Load Classifier
GRID_SEARCH = joblib.load('meerkat/binary_classifier/models/final_card.pkl')
