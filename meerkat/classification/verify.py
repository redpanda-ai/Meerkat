#!/usr/local/bin/python3.3

"""A script that tests the accuracy of a previously generated model using
a provided human labeled file

Created on Mar 11, 2014
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# Note: Transactions file must be pipe delimited.

# python3.3 -m meerkat.classification.verify [pickled_classifier] [labeled_transactions_file]
# python3.3 -m meerkat.classification.verify meerkat/classification/models/final_card.pkl data/input/users.txt

#####################################################

#import os
import sys

from sklearn.externals import joblib

from meerkat.classification.train import load_data

def test_model(file_to_test):
	"""Tests a classifier model using the provided file."""
	transactions, labels = load_data([], [], file_to_test)
	grid_search = joblib.load(sys.argv[1])
	score = grid_search.score(transactions, labels)
	print("Score: ", score)

def verify_arguments(arguments):
	""" Verifies proper usage """
	sufficient_arguments = (len(arguments) == 3)

	if not sufficient_arguments:
		print("Insufficient arguments. Please see usage")
		sys.exit()

	classifier = arguments[1]
	transactions_file = arguments[2]

	classifier_included = classifier.endswith('.pkl')
	transactions_included = transactions_file.endswith(".txt")

	if not classifier_included or not transactions_included:
		print("Erroneous arguments. Please see usage")
		sys.exit()

def run_from_command_line(arguments):
	"""Runs these commands if the module is invoked from the command line"""
	verify_arguments(arguments)
	test_model(arguments[2])

if __name__ == "__main__":
	run_from_command_line(sys.argv)
