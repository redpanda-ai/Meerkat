#!/usr/local/bin/python3
# pylint: disable=all

from sklearn.externals import joblib

def predict_if_physical_transaction(description=None):

	result = list(GRID_SEARCH.predict([description]))[0]

	return result

# Load Classifier
GRID_SEARCH = joblib.load('longtail/binary_classifier/US.pkl')