#!/usr/local/bin/python3
# pylint: disable=all

"""This module is a utility tool. It takes a file of 
transactions with a column header of DESCRIPTION_UNMASKED 
and outputs a file containing either only physical 
transactions or only non physical transactions
depending on a provided argument. This is useful in
terms of evaluating how well we deal with both
classes of transactions

Created on May 27, 2014
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# Note: Transactions file must be pipe delimited.

# python3.3 -m meerkat.binary_classifier.filter [pickled_classifier] [category_to_return] [transactions_file]
# python3.3 -m meerkat.binary_classifier.filter meerkat/binary_classifier/models/final_card.pkl 1 data/input/users.txt

#####################################################

import sys
from os.path import basename, splitext

from sklearn.externals import joblib

from meerkat.various_tools import load_dict_list, write_dict_list

def filter_transactions(transactions):
	""" Takes a list of transactions and only returns
	the desired category """

	filtered_transactions = []
	classifier = joblib.load(sys.argv[1])
	desired_category = sys.argv[2]

	print("Classifying, please wait \n")

	for i, transaction in enumerate(transactions):

		# Progress
		progress = (i / len(transactions)) * 100
		progress = str(round(progress, 2)) + "%"
		sys.stdout.write('\r')
		sys.stdout.write(progress)
		sys.stdout.flush()

		prediction = classify(classifier, transaction["DESCRIPTION_UNMASKED"])

		if prediction == desired_category:
			filtered_transactions.append(transaction)

	return filtered_transactions

def classify(classifier, description):

	"""This method uses a previously generated classifier to
	classify a single transaction as either physical 
	or non physical"""

	result = list(classifier.predict([description]))[0]
	#print(description, " : ", result)

	return result

def verify_arguments():
	""" Verifies proper usage """

	sufficient_arguments = (len(sys.argv) == 4)

	if not sufficient_arguments:
		print("Insufficient arguments. Please see usage")
		sys.exit()

	classifier = sys.argv[1]
	category = sys.argv[2]
	transactions_file = sys.argv[3]

	classifier_included = classifier.endswith('.pkl')
	transactions_included = transactions_file.endswith(".txt")
	supported_category = (category == "1" or category == "0")

	if not classifier_included or not transactions_included or not supported_category:
		print("Erroneous arguments. Please see usage")
		sys.exit()

def run_from_command_line(command_line_arguments):
	"""Runs these commands if the module is invoked from the command line"""

	verify_arguments()
	print("Loading file, please wait \n")
	transactions = load_dict_list(command_line_arguments[3])
	filtered_transactions = filter_transactions(transactions)
	basepath = splitext(basename(command_line_arguments[3]))[0]
	category = command_line_arguments[2]
	file_suffix = "_non_physical.txt" if category == "0" else "_physical.txt"
	output_folder = "/mnt/ephemeral/training_data/"
	output_file = output_folder + basepath + file_suffix
	print("Saving file, please wait \n")
	write_dict_list(filtered_transactions, output_file)
		
if __name__ == "__main__":
	run_from_command_line(sys.argv)
