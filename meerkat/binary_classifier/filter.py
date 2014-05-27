#!/usr/local/bin/python3
# pylint: disable=all

"""This module is a utility tool. It takes a file of 
transactions with a column header of DESCRIPTION and 
outputs a file containing either only physical 
transactions or only non physical transactions
depending on a provided argument. This is useful in
terms of evaluating how well we deal with both
classes of transactions"""

#################### USAGE ##########################

# Note: Transactions file must be pipe delimited.

# python3.3 -m meerkat.binary_classifier.filter [pickled_model] [category_to_return] [transactions_file]
# python3.3 -m meerkat.binary_classifier.filter meerkat/binary_classifier/models/final_card.pkl 1 data/input/users.txt

#####################################################

import sys

def verify_arguments():
	""" Verifies proper usage """

	sufficient_arguments = (len(sys.argv) == 4)

	if not sufficient_arguments:
		print("Insufficient arguments. Please see usage")
		sys.exit()

	model = sys.argv[1]
	category = sys.argv[2]
	transactions_file = sys.argv[3]

	model_included = model.endswith('.pkl')
	transactions_included = transactions_file.endswith(".txt")
	supported_category = (category == "1" or category == "0")

	if not model_included or not transactions_included or not supported_category:
		print("Erroneous arguments. Please see usage")
		sys.exit()
		
if __name__ == "__main__":

	verify_arguments()
