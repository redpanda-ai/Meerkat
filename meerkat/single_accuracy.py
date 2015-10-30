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
import logging
import sys
import random
import json
import statistics

from itertools import zip_longest

from meerkat.various_tools import load_dict_list, load_params
from meerkat.classification.lua_bridge import get_cnn_by_path
import argparse

parser = argparse.ArgumentParser(description="Test a given machine learning model against the given labeled data")
parser.add_argument('--testfile', '-f', required=True, help="path to the test data")
parser.add_argument('--model', '-m', required=True, help="path to the model under test")
parser.add_argument('--dictionary', '-d', required=True, help="mapping of model output IDs to human readable names")
parser.add_argument('--samples', '-s', required=False, help="optional number of random data points to test.  Default is all test data")
parser.add_argument('--humandictionary', '-D', required=True, help="Mapping of GOOD_DESCRIPTION names to the IDs output by your CNN")

def grouper(iterable):
	"""Returns batches of size 128 of iterable elements"""
	return zip_longest(*[iter(iterable)]*128, fillvalue={"DESCRIPTION_UNMASKED": ""})

def generic_test(machine, human, cnn_column, human_column):
	"""Tests both the recall and precision of the pinpoint classifier against
	human-labeled training data."""

	sys.stdout.write('\n')
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
		key = str(machine_row["UNIQUE_TRANSACTION_ID"])
		human_row = index_lookup.get(key)

		# Identify unlabeled points
		if not human_row or not human_row.get(human_column):
			needs_hand_labeling.append(machine_row[doc_label])
			continue

		if machine_row[cnn_column] == human_row[human_column]:
			correct.append(human_row[doc_label] +
			" (ACTUAL:" + human_row[human_column] + ")")
			continue

		mislabeled.append(human_row[doc_label] + " (ACTUAL: " + human_row[human_column] + ")" + " (FOUND: " + machine_row[cnn_column] + ")")

	return {
		"needs_hand_labeling": needs_hand_labeling,
		"mislabeled": mislabeled,
		"unlabeled": unlabeled,
		"correct": correct
	}

def vest_accuracy(transactions, label_key, result_list):
	"""Takes file by default but can accept result
	queue/ non_physical list. Attempts to provide various
	accuracy tests"""

	machine_labeled = result_list or []
	if len(machine_labeled) <= 0:
		logging.warning("No labeled results provided to vest_accuracy()")
		return

	# Load Verification Source
	human_labeled = transactions or []
	if len(human_labeled) <= 0:
		logging.warning("No human labeled transactions provided to vest_accuracy()")

	# Test Classifier for recall and precision
	acc_results = generic_test(machine_labeled, human_labeled, label_key, "GOOD_DESCRIPTION")

	# Collect results into dict for easier access
	total_processed = len(machine_labeled)
	num_labeled = total_processed - len(acc_results["unlabeled"])
	num_verified = num_labeled - len(acc_results["needs_hand_labeling"])
	return {
		"total_processed": total_processed,
		"correct": acc_results["correct"],
		"needs_hand_labeling": acc_results["needs_hand_labeling"],
		"unlabeled": acc_results["unlabeled"],
		"num_verified": num_verified,
		"num_labeled": num_labeled,
		"mislabeled": acc_results["mislabeled"],
		"total_recall": num_labeled / total_processed * 100,
		"precision": len(acc_results["correct"]) / max(num_verified, 1) * 100
	}

def apply_cnn(classifier, transactions):
	"""Apply the CNN to transactions"""

	batches = grouper(transactions)
	processed = []

	for i, batch in enumerate(batches):
		processed += classifier(batch, doc_key="DESCRIPTION_UNMASKED", label_key="MERCHANT_NAME")

	return processed[0:len(transactions)]

def CNN_accuracy(test_file, model, model_dict, num_tests, human_dict):
	"""Run given CNN on a file of Merchant Samples"""

	# Load Classifier, and transactions
	classifier = get_cnn_by_path(model, model_dict)
	transactions = load_dict_list(test_file)
	transactions = num_tests and random.sample(transactions, min(num_tests, len(transactions))) or transactions

	# Label the points using the classifier and report accuracy
	labeled_trans = apply_cnn(classifier, transactions)
	human_map = load_params(human_dict)
	machine_map = load_params(model_dict)
	for trans in labeled_trans:
		if trans["GOOD_DESCRIPTION"].lower() in human_map:
			trans["GOOD_DESCRIPTION"] = machine_map[str(human_map[trans["GOOD_DESCRIPTION"].lower()])]
	accuracy_results = vest_accuracy(transactions, "MERCHANT_NAME", labeled_trans)

	print_results(accuracy_results)
	# results = open("data/output/single_test.csv", "a")
	# writer = csv.writer(results, delimiter=',', quotechar='"')
	# writer.writerow([merchant["name"], merchant['total_recall'], merchant["precision"]])
	# results.close()

def print_results(results):
	"""Provide useful readable output"""

	if results is None:
		return

	sys.stdout.write('\n\n')

	print("STATS:")
	print("{0:35} = {1:11}".format("Total Transactions Processed",
		results['total_processed']))

	sys.stdout.write('\n')

	print("{0:35} = {1:10.2f}%".format("Recall all transactions",
		results['total_recall']))
	print("{0:35} = {1:11}".format("Number of transactions labeled",
		results['num_labeled']))
	print("{0:35} = {1:11}".format("Number of transactions verified",
		results['num_verified']))
	print("{0:35} = {1:10.2f}%".format("Precision",
		results['precision']))

def run_from_command_line(args):
	"""Runs these commands if the module is invoked from the command line"""

	CNN_accuracy(args.testfile, args.model, args.dictionary, args.samples, args.humandictionary)

if __name__ == "__main__":
	cmd_args = parser.parse_args()
	run_from_command_line(cmd_args)
