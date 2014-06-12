#!/usr/local/bin/python3.3

"""A script that tests the accuracy of a previously generated
model using a provided human labeled file

Created on Mar 11, 2014
@author: Matthew Sevrens
"""

import os
import sys

from sklearn.externals import joblib

from meerkat.binary_classifier.train import load_data

def test_model(file_to_test):
	"""Tests a classifier model using the provided file."""
	transactions, labels = load_data([], [], file_to_test)
	grid_search = joblib.load('meerkat/binary_classifier/final_card.pkl')
	score = grid_search.score(transactions, labels)
	print("Score: ", score)

if __name__ == "__main__":
	if len(sys.argv) == 2 and os.path.isfile(sys.argv[1]):
		test_model(sys.argv[1])
	else:
		test_model("data/misc/verifiedLabeledTrans.csv")
