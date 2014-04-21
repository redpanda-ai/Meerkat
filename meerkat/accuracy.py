#!/usr/local/bin/python3

"""This script tests the current accuracy of our labeling tool"""

import csv
import datetime
import logging
import os
import sys
from pprint import pprint
from meerkat.various_tools import load_dict_list


def test_pinpoint_classifier(machine_labeled, human_labeled, my_lists):
	"""Tests both the recall and precision of the pinpoint classifier against
	human-labeled training data."""
	for machine_labeled_row in machine_labeled:
		# Our confidence was not high enough to label
		if machine_labeled_row["factual_id"] == "":
			my_lists["unlabeled"].append(machine_labeled_row['DESCRIPTION'])
			continue
		# Verify against human labeled
		for index, human_labeled_row in enumerate(human_labeled):
			if machine_labeled_row['DESCRIPTION'] == human_labeled_row['DESCRIPTION']:
				if human_labeled_row['IS_PHYSICAL_TRANSACTION'] == '0':
					# Transaction is non physical
					my_lists["non_physical"].append(machine_labeled_row['DESCRIPTION'])
					break
				elif human_labeled_row["factual_id"] == "":
					# Transaction is not yet labeled
					my_lists["needs_hand_labeling"].append(machine_labeled_row['DESCRIPTION'])
					break
				elif machine_labeled_row["factual_id"] == human_labeled_row["factual_id"]:
					# Transaction was correctly labeled
					my_lists["correct"].append(human_labeled_row['DESCRIPTION']\
						+ " (ACTUAL:" + human_labeled_row["factual_id"] + ")")
					break
				else:
					# Transaction is mislabeled
					my_lists["mislabeled"].append(human_labeled_row['DESCRIPTION']\
						+ " (ACTUAL:" + human_labeled_row["factual_id"] + ")"\
						+ " (FOUND:" + machine_labeled_row["factual_id"] + ")")
					break
			elif index + 1 == len(human_labeled):
				my_lists["needs_hand_labeling"].append(machine_labeled_row['DESCRIPTION'])

def test_bulk_classifier(human_labeled, non_physical_trans, my_lists):
	"""Tests for accuracy of the bulk (binary) classifier"""
	for item in my_lists["unlabeled"]:
		for _, human_labeled_row in enumerate(human_labeled):
			if item == human_labeled_row['DESCRIPTION']:
				if human_labeled_row['IS_PHYSICAL_TRANSACTION'] == '0':
					# Transaction is non physical
					my_lists["non_physical"].append(item)
					break

	for item in non_physical_trans:
		for _, human_labeled_row in enumerate(human_labeled):
			if item == human_labeled_row['DESCRIPTION']:
				if human_labeled_row['IS_PHYSICAL_TRANSACTION'] == '1':
					my_lists["incorrect_non_physical"].append(item)

def test_accuracy(params, file_path=None, non_physical_trans=None,
	result_list=None):
	"""Takes file by default but can accept result
	queue/ non_physical list. Attempts to provide various
	accuracy tests"""

	if non_physical_trans is None:
		non_physical_trans = []
	if result_list is None:
		result_list = []

	if len(result_list) > 0:
		machine_labeled = result_list
	elif file_path is not None and os.path.isfile(file_path):
		machine_labeled_file = open(file_path, encoding="utf-8", errors='replace')
		machine_labeled = list(csv.DictReader(machine_labeled_file))
	else:
		logging.warning("Not enough information provided to perform "\
			+ "accuracy tests on")
		return

	# Load Verification Source
	verification_source = params.get("verification_source",
		"data/misc/verifiedLabeledTrans.csv")
	human_labeled = load_dict_list(verification_source, delimiter=",")

	my_counters = {
		"total": len(machine_labeled),
		"total_processed": len(machine_labeled) + len(non_physical_trans)
	}

	#Abort if there is nothing to process
	if my_counters["total"] == 0 or my_counters["total_processed"] == 0:
		logging.warning("Nothing provided to perform accuracy tests on")
		return

	my_lists = {
		"needs_hand_labeling": [], "non_physical": [], "mislabeled": [],
		"unlabeled": [], "correct": [], "incorrect_non_physical" : []
	}

	# Test Pinpoint Classifier for recall and precision
	test_pinpoint_classifier(machine_labeled, human_labeled, my_lists)

	# Test Bulk (binary) Classifier for accuracy
	test_bulk_classifier(human_labeled, non_physical_trans, my_lists)

	# Collect results into dict for easier access
	my_counters["num_labeled"] = my_counters["total"] - len(my_lists["unlabeled"])
	my_counters["num_verified"] = my_counters["num_labeled"] -\
		len(my_lists["needs_hand_labeling"])
	if my_counters["num_verified"] <= 0:
		my_counters["num_verified"] = 1
	binary_accuracy = 100 - ((len(my_lists["non_physical"])\
		+ len(my_lists["incorrect_non_physical"])) /
		my_counters["total_processed"]) * 100

	#rounded_percent = lambda x: math.ceil(x * 100)

	return {
		"total_processed": my_counters["total_processed"],
		"total_physical": my_counters["total"] / my_counters["total_processed"] * 100,
		"total_non_physical": len(non_physical_trans) /
			my_counters["total_processed"] * 100,
		"correct": my_lists["correct"],
		"needs_hand_labeling": my_lists["needs_hand_labeling"],
		"non_physical": my_lists["non_physical"],
		"unlabeled": my_lists["unlabeled"],
		"num_verified": my_counters["num_verified"],
		"num_labeled": my_counters["num_labeled"],
		"mislabeled": my_lists["mislabeled"],
		"total_recall": my_counters["num_labeled"] /
			my_counters["total_processed"] * 100,
		"total_recall_physical": my_counters["num_labeled"] /
			my_counters["total"] * 100,
		"precision": len(my_lists["correct"]) / my_counters["num_verified"] * 100,
		"binary_accuracy": binary_accuracy
	}

def speed_tests(start_time, accuracy_results):
	"""Run a number of tests related to speed"""

	time_delta = datetime.datetime.now() - start_time
	seconds = time_delta.seconds if time_delta.seconds > 0 else 1

	time_per_transaction = seconds / accuracy_results['total_processed']
	transactions_per_minute = (accuracy_results['total_processed'] / seconds) * 60

	print("\nSPEED TESTS:")
	print("{0:35} = {1:11}".format("Total Time Taken", str(time_delta)[0:11]))
	print("{0:35} = {1:11.2f}".format("Time per Transaction (in seconds)",
		time_per_transaction))
	print("{0:35} = {1:11.2f}".format("Transactions Per Minute",
		transactions_per_minute))

	return {'time_delta':time_delta,
			'time_per_transaction': time_per_transaction,
			'transactions_per_minute':transactions_per_minute}

def print_results(results):
	"""Provide useful readable output"""

	if results is None:
		return

	print("\nSTATS:")
	print("{0:35} = {1:11}".format("Total Transactions Processed",
		results['total_processed']))
	print("{0:35} = {1:10.2f}%".format("Total Labeled Physical",
		results['total_physical']))
	print("{0:35} = {1:10.2f}%".format("Total Labeled Non Physical",
		results['total_non_physical']))
	print("{0:35} = {1:10.2f}%".format("Binary Classifier Accuracy",
		results['binary_accuracy']))
	print("\n")
	print("{0:35} = {1:10.2f}%".format("Recall all transactions",
		results['total_recall']))
	print("{0:35} = {1:10.2f}%".format("Recall physical",
		results['total_recall_physical']))
	print("{0:35} = {1:11}".format("Number of transactions labeled",
		results['num_labeled']))
	print("{0:35} = {1:11}".format("Number of transactions verified",
		results['num_verified']))
	print("{0:35} = {1:10.2f}%".format("Precision",
		results['precision']))
	print("", "MISLABELED:", '\n'.join(results['mislabeled']), sep="\n")
	print("", "MISLABELED BINARY:", '\n'.join(results['non_physical']),
		sep="\n")

def run_from_command_line(command_line_arguments):
	"""Runs these commands if the module is invoked from the command line"""
	output_path = "data/output/meerkatLabeled.csv"
	if len(command_line_arguments) > 1:
		output_path = command_line_arguments[1]
	pprint(test_accuracy(params=None, file_path=output_path))

if __name__ == "__main__":
	run_from_command_line(sys.argv)
