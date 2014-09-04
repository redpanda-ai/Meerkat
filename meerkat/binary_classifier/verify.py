#!/usr/local/bin/python3.3

"""A script that tests the accuracy of a previously generated model using
a provided human labeled file

Created on Mar 11, 2014
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# Note: Transactions file must be pipe delimited.

# python3.3 -m meerkat.binary_classifier.verify [pickled_classifier] [labeled_transactions_file]
# python3.3 -m meerkat.binary_classifier.verify meerkat/binary_classifier/models/final_card.pkl data/input/users.txt

#####################################################

import os
import sys

from sklearn.externals import joblib

from meerkat.binary_classifier.train import load_data

def test_model(file_to_test):
	"""Tests a classifier model using the provided file."""
	transactions, labels = load_data([], [], file_to_test)
	grid_search = joblib.load(sys.argv[1])
	score = grid_search.score(transactions, labels)
	print("Score: ", score)

def verify_arguments():
	""" Verifies proper usage """

	sufficient_arguments = (len(sys.argv) == 3)

	if not sufficient_arguments:
		print("Insufficient arguments. Please see usage")
		sys.exit()

	classifier = sys.argv[1]
	transactions_file = sys.argv[2]

	classifier_included = classifier.endswith('.pkl')
	transactions_included = transactions_file.endswith(".txt")

	if not classifier_included or not transactions_included:
		print("Erroneous arguments. Please see usage")
		sys.exit()

def run_from_command_line(command_line_arguments):
	"""Runs these commands if the module is invoked from the command line"""

	verify_arguments()
	test_model(sys.argv[2])

if __name__ == "__main__":
	run_from_command_line(sys.argv)
