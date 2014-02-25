#!/usr/local/bin/python3
# pylint: disable=all

from sklearn.externals import joblib

def predict_if_physical_transaction(description=None):

	grid_search = joblib.load('longtail/binary_classifier/global.pkl')
	result = list(grid_search.predict([description]))[0]

	return result