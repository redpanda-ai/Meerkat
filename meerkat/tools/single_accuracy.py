#!/usr/local/bin/python3.3

"""This utility loads a single classifier, runs it over the given data, compares the classifier's answers to the

Created in October, 2015
@author: Matthew Sevrens
@author: J. Andrew Key
@author: Joseph Altmaier
"""

#################### USAGE ##########################

# python3 -m meerkat.tools.single_accuracy -m PATH_TO_CLASSIFIER -d PATH_TO_CLASSIFIER_OUTPUT_MAP -D PATH_TO_TEST_DATA_NAME_MAPPING -f PATH_TO_TEST_DATA

#####################################################

import csv
import sys
import json
import pandas as pd

from meerkat.various_tools import load_params
from meerkat.classification.lua_bridge import get_cnn_by_path
import argparse

parser = argparse.ArgumentParser(description="Test a given machine learning model against the given labeled data")
parser.add_argument('--testfile', '-f', required=True, help="path to the test data")
parser.add_argument('--model', '-m', required=True, help="path to the model under test")
parser.add_argument('--dictionary', '-d', required=True, help="mapping of model output IDs to human readable names")
parser.add_argument('--humandictionary', '-D', required=False, help="Optional mapping of GOOD_DESCRIPTION names to the IDs output by your CNN")

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

		if machine_map and human_map and human_row[human_column].lower() in human_map:
			human_row[human_column] = machine_map[str(human_map[human_row[human_column].lower()])]

		if machine_row[cnn_column] == human_row[human_column]:
			correct.append(human_row[doc_label] +
			" (ACTUAL:" + human_row[human_column] + ")")
			continue

		mislabeled.append(human_row[doc_label] + " (ACTUAL: " + human_row[human_column] + ")" + " (FOUND: " + machine_row[cnn_column] + ")")

	return len(machine), needs_hand_labeling, mislabeled, unlabeled, correct

def CNN_accuracy(test_file, classifier, model_dict=None, human_dict=None):
	"""Run given CNN on a file of Merchant Samples"""
	# Load Classifier, and transactions
	human_map = __load_label_map(human_dict)
	machine_map = __load_label_map(model_dict)
	reader = pd.read_csv(test_file, chunksize=1000, na_filter=False, quoting=csv.QUOTE_NONE, encoding="utf-8", sep='|', error_bad_lines=False)

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

	num_labeled = bulk_total - bulk_unlabeled
	num_verified = num_labeled - bulk_needs_hand_labeling
	total_recall = num_labeled / bulk_total * 100
	precision = bulk_correct / max(num_verified, 1) * 100
	return bulk_total, bulk_needs_hand_labeling, bulk_correct, bulk_mislabeled, bulk_unlabeled, num_labeled, num_verified, total_recall, precision
	# results = open("data/output/single_test.csv", "a")
	# writer = csv.writer(results, delimiter=',', quotechar='"')
	# writer.writerow([merchant["name"], merchant['total_recall'], merchant["precision"]])
	# results.close()

def __load_label_map(label_map):
	if isinstance(label_map, dict):
		return label_map
	return label_map and load_params(label_map) or None

def print_results(total, needs_hand_labeling, correct, mislabeled, unlabeled, num_labeled, num_verified, total_recall, precision):
	"""Provide useful readable output"""

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

	classifier = get_cnn_by_path(args.model, args.dictionary)
	print_results(*CNN_accuracy(args.testfile, classifier, args.dictionary, args.humandictionary))

if __name__ == "__main__":
	cmd_args = parser.parse_args()
	run_from_command_line(cmd_args)
