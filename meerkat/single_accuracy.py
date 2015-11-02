#!/usr/local/bin/python3.3

"""This script is aimed at testing the accuracy of the Meerkat Classifier.
We iteratively use this accuracy as a feedback score to tune and optimize
Meerkat as a whole. The use of this script requires a csv containing human
labeled data that has been verified as accurate. The input file to this
script should be randomized over the selection of data being modeled.

To configure accuracy, this module should be provided a verification
source that can be referenced in the config file under the key
params["verification_source"]. This should be a reference to a file
containing manually labeled transactions.

Created on Jan 8, 2014
@author: Matthew Sevrens
@author: J. Andrew Key
"""

#################### USAGE ##########################

# Note: Needs refining. Experts only!

# Suggested Future Work:
#
# 	Instead of matching being binary, produce a
# 	loss function that penalizes less for matching
# 	the wrong location if the majority of the
#	informaiton is correct

#####################################################

import csv
import sys
import json
import pandas as pd
import time

from meerkat.various_tools import load_params
from meerkat.classification.lua_bridge import get_cnn_by_path
import argparse

parser = argparse.ArgumentParser(description="Test a given machine learning model against the given labeled data")
parser.add_argument('--testfile', '-f', required=True, help="path to the test data")
parser.add_argument('--model', '-m', required=True, help="path to the model under test")
parser.add_argument('--dictionary', '-d', required=True, help="mapping of model output IDs to human readable names")
parser.add_argument('--humandictionary', '-D', required=True, help="Mapping of GOOD_DESCRIPTION names to the IDs output by your CNN")

def generic_test(machine, human, cnn_column, human_column, human_map, machine_map):
	"""Tests both the recall and precision of the pinpoint classifier against
	human-labeled training data."""
	doc_label = 'DESCRIPTION_UNMASKED'

	# Create Quicker Lookup
	index_lookup = {row["UNIQUE_TRANSACTION_ID"]: row for row in human}

	unlabeled = []
	needs_hand_labeling = []
	correct = []
	mislabeled = []
	# Test Each Machine Labeled Row
	for index, machine_row in enumerate(machine):
		# Continue if Unlabeled
		if machine_row[cnn_column] == "":
			unlabeled.append(machine_row[doc_label])
			continue

		# Get human answer index
		key = machine_row["UNIQUE_TRANSACTION_ID"]
		human_row = index_lookup.get(key)

		# Identify unlabeled points
		if not human_row or not human_row.get(human_column):
			needs_hand_labeling.append(machine_row[doc_label])
			continue

		if human_row[human_column].lower() in human_map:
			human_row[human_column] = machine_map[str(human_map[human_row[human_column].lower()])]

		if machine_row[cnn_column] == human_row[human_column]:
			correct.append(human_row[doc_label] +
			" (ACTUAL:" + human_row[human_column] + ")")
			continue

		mislabeled.append(human_row[doc_label] + " (ACTUAL: " + human_row[human_column] + ")" + " (FOUND: " + machine_row[cnn_column] + ")")

	return len(machine), needs_hand_labeling, mislabeled, unlabeled, correct

def CNN_accuracy(test_file, model, model_dict, human_dict):
	"""Run given CNN on a file of Merchant Samples"""
	start = time.time()

	# Load Classifier, and transactions
	classifier = get_cnn_by_path(model, model_dict)
	human_map = load_params(human_dict)
	machine_map = load_params(model_dict)
	reader = pd.read_csv(test_file, chunksize=1000, na_filter=False,\
	quoting=csv.QUOTE_NONE, encoding="utf-8", sep='|', error_bad_lines=False)

	bulk_total = 0
	bulk_needs_hand_labeling = 0
	bulk_mislabeled = 0
	bulk_unlabeled = 0
	bulk_correct = 0
	# Process Transactions
	for chunk in reader:
		transactions = chunk.to_dict("records")
		# Label the points using the classifier and report accuracy
		machine_labeled = classifier(transactions, doc_key="DESCRIPTION_UNMASKED", label_key="MERCHANT_NAME")

		total, needs_hand_labeling, mislabeled, unlabeled, correct = generic_test(machine_labeled, transactions, "MERCHANT_NAME", "GOOD_DESCRIPTION", human_map, machine_map)
		bulk_total += total
		bulk_needs_hand_labeling += len(needs_hand_labeling)
		bulk_mislabeled += len(mislabeled)
		bulk_unlabeled += len(unlabeled)
		bulk_correct += len(correct)

	print_results(bulk_total, bulk_needs_hand_labeling, bulk_correct, bulk_mislabeled, bulk_unlabeled)
	print("Total run time was {}s".format(time.time() - start))
	# results = open("data/output/single_test.csv", "a")
	# writer = csv.writer(results, delimiter=',', quotechar='"')
	# writer.writerow([merchant["name"], merchant['total_recall'], merchant["precision"]])
	# results.close()

def print_results(total, needs_hand_labeling, correct, mislabeled, unlabeled):
	"""Provide useful readable output"""
	num_labeled = total - unlabeled
	num_verified = num_labeled - needs_hand_labeling
	total_recall = num_labeled / total * 100
	precision = correct / max(num_verified, 1) * 100

	sys.stdout.write('\n\n')

	print("STATS:")
	print("{0:35} = {1:11}".format("Total Transactions Processed", total))

	sys.stdout.write('\n')

	print("{0:35} = {1:10.2f}%".format("Recall all transactions", total_recall))
	print("{0:35} = {1:11}".format("Number of transactions labeled", num_labeled))
	print("{0:35} = {1:11}".format("Number of transactions verified", num_verified))
	print("{0:35} = {1:10.2f}%".format("Precision", precision))

def run_from_command_line(args):
	"""Runs these commands if the module is invoked from the command line"""

	CNN_accuracy(args.testfile, args.model, args.dictionary, args.humandictionary)

if __name__ == "__main__":
	cmd_args = parser.parse_args()
	run_from_command_line(cmd_args)
