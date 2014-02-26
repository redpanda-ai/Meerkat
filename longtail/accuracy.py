#!/usr/local/bin/python3
# pylint: disable=C0103
# pylint: disable=C0301

"""This script tests the current accuracy of our labeling tool"""

import csv
import datetime
import logging
import math
import os
import sys

def test_accuracy(file_path=None, non_physical_trans=[], result_list=[]):
	"""Takes file by default but can accept result
	queue/ non_physical list. Attempts to provide various
	accuracy tests"""

	if len(result_list) > 0:
		machine_labeled = result_list
	elif file_path is not None and os.path.isfile(file_path):
		ML_file = open(file_path)
		machine_labeled = list(csv.DictReader(ML_file))
	else:
		logging.warning("Not enough information provided to perform accuracy tests on")
		return

	#FIXME: Hard-coded?
	human_labeled_input_file = open("data/misc/verifiedLabeledTrans.factual.csv")
	human_labeled = list(csv.DictReader(human_labeled_input_file))
	human_labeled_input_file.close()

	#Ensure there is something to process
	total = len(machine_labeled)
	total_processed = len(machine_labeled) + len(non_physical_trans)

	if total == 0 or total_processed == 0:
		logging.warning("Nothing provided to perform accuracy tests on")
		return

	non_physical_trans = non_physical_trans or []
	needs_hand_labeling = []
	non_physical = []
	mislabeled = []
	unlabeled = []
	correct = []

	# Test Recall / Precision
	for machine_labeled_row in machine_labeled:

		# Our confidence was not high enough to label
		if machine_labeled_row['factual_id'] == "":
			unlabeled.append(machine_labeled_row['DESCRIPTION'])
			continue

		# Verify against human labeled
		for index, human_labeled_row in enumerate(human_labeled):
			if machine_labeled_row['DESCRIPTION'] == human_labeled_row['DESCRIPTION']:
				if human_labeled_row['factual_id'] == "":
					# Transaction is not yet labeled
					needs_hand_labeling.append(machine_labeled_row['DESCRIPTION'])
					break
				elif machine_labeled_row['factual_id'] == human_labeled_row['factual_id']:
					# Transaction was correctly labeled
					correct.append(human_labeled_row['DESCRIPTION'] + " (ACTUAL:" + human_labeled_row['factual_id'] + ")")
					break
				elif human_labeled_row['IS_PHYSICAL_TRANSACTION'] == '0':
					# Transaction is non physical
					non_physical.append(machine_labeled_row['DESCRIPTION'])
					break
				else:
					# Transaction is mislabeled
					mislabeled.append(human_labeled_row['DESCRIPTION'] + " (ACTUAL:" + human_labeled_row['factual_id'] + ")")
					break
			elif index + 1 == len(human_labeled):
				needs_hand_labeling.append(machine_labeled_row['DESCRIPTION'])

	# Test Binary
	for item in unlabeled:
		for index, human_labeled_row in enumerate(human_labeled):
			if item == human_labeled_row['DESCRIPTION']:
				if human_labeled_row['IS_PHYSICAL_TRANSACTION'] == '0':
					# Transaction is non physical
					non_physical.append(item)
					break

	# Collect results into dict for easier access
	num_labeled = total - len(unlabeled)
	num_verified = num_labeled - len(needs_hand_labeling)
	num_verified = num_verified if num_verified > 0 else 1
	num_correct = len(correct)

	results = {}
	results['total_processed'] = total_processed
	results['total_physical'] = math.ceil((len(machine_labeled) / results['total_processed']) * 100)
	results['total_non_physical'] = math.ceil((len(non_physical_trans) / results['total_processed']) * 100)
	results['correct'] = correct
	results['needs_hand_labeling'] = needs_hand_labeling
	results['non_physical'] = non_physical
	results['unlabeled'] = unlabeled
	results['num_verified'] = num_verified
	results['mislabeled'] = mislabeled
	results['total_recall'] = math.ceil((num_labeled / results['total_processed']) * 100)
	results['total_recall_non_physical'] = math.ceil((num_labeled / total) * 100)
	results['precision'] = math.ceil((num_correct / num_verified) * 100)
	results['binary_accuracy'] = 100 - math.ceil((len(non_physical) / total) * 100)

	return results

def speed_tests(start_time, accuracy_results):
	"""Run a number of tests related to speed"""

	time_delta = datetime.datetime.now() - start_time
	time_per_transaction = time_delta.seconds / accuracy_results['total_processed']
	transactions_per_minute = (accuracy_results['total_processed'] / time_delta.seconds) * 60

	print("\nSPEED TESTS:")
	print("{0:35} = {1:11}".format("Total Time Taken", str(time_delta)[0:10]))
	print("{0:35} = {1:11.2f}".format("Time per Transaction (in seconds)", time_per_transaction))
	print("{0:35} = {1:11.2f}".format("Transactions Per Minute", transactions_per_minute))

	return {'time_delta':time_delta,
			'time_per_transaction': time_per_transaction,
			'transactions_per_minute':transactions_per_minute}

def print_results(results):
	"""Provide useful readable output"""

	if results is None:
		return
		
	print("\nSTATS:")
	print("{0:35} = {1:11}".format("Total Transactions Processed", results['total_processed']))
	print("{0:35} = {1:11}".format("Total Labeled Physical", results['total_physical']))
	print("{0:35} = {1:11.2f}%".format("Total Labeled Non Physical", results['total_non_physical']))
	print("{0:35} = {1:11.2f}%".format("Binary Classifier Accuracy", results['binary_accuracy']))
	print("\n")
	print("{0:35} = {1:11.2f}%".format("Recall all transactions", results['total_recall']))
	print("{0:35} = {1:11.2f}%".format("Recall non physical", results['total_recall_non_physical']))
	print("{0:35} = {1:11}".format("Number of transactions verified", results['num_verified']))
	print("{0:35} = {1:11.2f}%".format("Precision", results['precision']))
	print("", "MISLABELED:", '\n'.join(results['mislabeled']), sep="\n")

if __name__ == "__main__":
	output_path = sys.argv[1] if len(sys.argv) > 1 else "data/output/longtailLabeled.csv"
	print_results(test_accuracy(file_path=output_path))
