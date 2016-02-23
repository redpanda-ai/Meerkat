#!/usr/local/bin/python3.3
# pylint: disable=line-too-long,invalid-name

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
import datetime
import logging
import os
import sys
import contextlib
import boto

from boto.s3.connection import Location

import pandas as pd

from meerkat.various_tools import load_dict_list, safely_remove_file, load_params
from meerkat.classification.lua_bridge import get_cnn

default_doc_key = "DESCRIPTION_UNMASKED"
default_label_key = "GOOD_DESCRIPTION"

class DummyFile(object):
	"""Resemble the stdout/stderr object but it prints nothing to screen"""
	def write(self, msg):
		"""It writes nothing, on purpose"""
		pass

@contextlib.contextmanager
def nostdout():
	"""
	It redirects the stderr stream to DummyFile object that do nothing with error message.
	'yield' is where unit tests take place.
	After the yield, restore sys.stderr and stdout to its original structure
	"""
	save_stdout = sys.stdout
	save_stderr = sys.stderr
	sys.stdout = DummyFile()
	sys.stderr = DummyFile()
	yield
	sys.stderr = save_stderr
	sys.stdout = save_stdout

def get_s3_connection():
	"""Returns a connection to S3"""
	try:
		conn = boto.connect_s3()
	except boto.s3.connection.HostRequiredError:
		print("Error connecting to S3, check your credentials")
		sys.exit()
	return conn

def test_bulk_classifier(human_labeled, non_physical_trans, my_lists):
	"""Tests for accuracy of the bulk (binary) classifier"""
	for item in my_lists["unlabeled"]:
		for _, human_labeled_row in enumerate(human_labeled):
			if item == human_labeled_row['DESCRIPTION_UNMASKED']:
				if human_labeled_row['IS_PHYSICAL_TRANSACTION'] == '0':
					# Transaction is non physical
					my_lists["non_physical"].append(item)
					break

	for item in non_physical_trans:
		for _, human_labeled_row in enumerate(human_labeled):
			if item == human_labeled_row['DESCRIPTION_UNMASKED']:
				if human_labeled_row['IS_PHYSICAL_TRANSACTION'] == '1':
					my_lists["incorrect_non_physical"].append(item)

def generic_test(*args, **kwargs):
	"""Finds both the percent labeled and predictive accuracy of the pinpoint classifier against
	human-labeled training data."""
	machine, human, cnn_column, human_column, human_map, machine_map = args[:]	
	doc_key = kwargs.get("doc_key", default_doc_key)

	# Create Quicker Lookup
	index_lookup = {row["UNIQUE_TRANSACTION_ID"]: row for row in human}

	unlabeled = []
	needs_hand_labeling = []
	correct = []
	mislabeled = []
	# Test Each Machine Labeled Row
	for _, machine_row in enumerate(machine):
		# Continue if Unlabeled
		if machine_row[cnn_column] == "":
			unlabeled.append(machine_row[doc_key])
			continue

		# Get human answer index
		key = machine_row["UNIQUE_TRANSACTION_ID"]
		human_row = index_lookup.get(key)

		# Identify unlabeled points
		if not human_row or not human_row.get(human_column):
			needs_hand_labeling.append(machine_row[doc_key])
			continue

		if machine_map and human_map and human_row[human_column].lower() in human_map:
			human_row[human_column] = machine_map[str(human_map[human_row[human_column].lower()])]

		if machine_row[cnn_column] == human_row[human_column]:
			correct.append(human_row[doc_key] +
			" (ACTUAL:" + human_row[human_column] + ")")
			continue

		mislabeled.append(human_row[doc_key] + " (ACTUAL: " + human_row[human_column] + ")" + " (FOUND: " + machine_row[cnn_column] + ")")

	return len(machine), needs_hand_labeling, mislabeled, unlabeled, correct

def CNN_accuracy(*args, **kwargs):
	"""Run given CNN over a given input file and return some stats"""
	# Load Classifier, and transactions
	test_file, classifier = args[:]
	model_dict = kwargs.get("model_dict", None)
	human_dict = kwargs.get("human_dict", None)
	label_key = kwargs.get("label_key", default_label_key)
	doc_key = kwargs.get("doc_key", default_doc_key)

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
		machine_labeled = classifier(transactions, doc_key=doc_key, label_key="CNN_output")

		total, needs_hand_labeling, mislabeled, unlabeled, correct = generic_test(machine_labeled, transactions, "CNN_output", label_key, human_map, machine_map, doc_key=doc_key)
		bulk_total += total
		bulk_needs_hand_labeling += len(needs_hand_labeling)
		bulk_mislabeled += len(mislabeled)
		bulk_unlabeled += len(unlabeled)
		bulk_correct += len(correct)

	return enhance_results(bulk_total, bulk_needs_hand_labeling, bulk_mislabeled, bulk_unlabeled, bulk_correct)
	# results = open("data/output/single_test.csv", "a")
	# writer = csv.writer(results, delimiter=',', quotechar='"')
	# writer.writerow([merchant["name"], merchant['percent_labeled'], merchant["predictive_accuracy"]])
	# results.close()

def __load_label_map(label_map):
	"""Provide label map"""
	if isinstance(label_map, dict):
		return label_map
	return label_map and load_params(label_map) or None

def print_results(results):
	"""Provide useful readable output"""

	sys.stdout.write('\n\n')

	print("STATS:")
	print("{0:35} = {1:11}".format("Total Transactions Processed", results["total"]))

	sys.stdout.write('\n')

	print("{0:35} = {1:10.2f}%".format("Percent of labeled transactions", results["percent_labeled"]))
	print("{0:35} = {1:11}".format("Number of transactions labeled", results["num_labeled"]))
	print("{0:35} = {1:11}".format("Number of transactions verified", results["num_verified"]))
	print("{0:35} = {1:10.2f}%".format("Predictive Accuracy", results["predictive_accuracy"]))

def enhance_results(total, needs_hand_labeling, mislabeled, unlabeled, correct):
	"""Group results"""

	num_labeled = total - unlabeled
	percent_labeled_physical = num_labeled / total * 100
	num_verified = num_labeled - needs_hand_labeling
	percent_labeled = num_labeled / total * 100
	predictive_accuracy = correct / max(num_verified, 1) * 100

	return {
		"total": total,
		"needs_hand_labeling": needs_hand_labeling,
		"mislabeled": mislabeled,
		"unlabeled": unlabeled,
		"correct": correct,
		"percent_labeled_physical": percent_labeled_physical,
		"num_labeled": num_labeled,
		"num_verified": num_verified,
		"percent_labeled": percent_labeled,
		"predictive_accuracy": predictive_accuracy
		}

def vest_accuracy(params, file_path=None, non_physical_trans=[], result_list=[]):
	"""Takes file by default but can accept result
	queue/ non_physical list. Attempts to provide various
	accuracy tests"""

	# Load machine labeled transactions
	if len(result_list) > 0:
		machine_labeled = result_list
	if not result_list and file_path and os.path.isfile(file_path):
		machine_labeled = load_dict_list(file_path)
	if not machine_labeled or len(machine_labeled) <= 0:
		logging.warning("Not enough information provided to perform " + "accuracy tests on")
		return

	# Load human labeled transactions
	verification_source = params.get("verification_source", "data/misc/verifiedLabeledTrans.txt")
	human_labeled = load_dict_list(verification_source)

	# Test Classifier for percent labeled and predictive accuracy
	label_key = params.get("label_key", "FACTUAL_ID")
	total, needs_hand_labeling, mislabeled, unlabeled, correct = generic_test(machine_labeled, human_labeled, label_key, label_key, None, None)
	results = enhance_results(total, len(needs_hand_labeling), len(mislabeled), len(unlabeled), len(correct))

	total_processed = len(machine_labeled) + len(non_physical_trans)
	results["total_processed"] = total_processed
	results["total_non_physical"] = len(non_physical_trans) / total_processed * 100
	results["total_physical"] = total / total_processed * 100
	return results

def speed_vests(start_time, accuracy_results):
	"""Run a number of tests related to speed"""

	time_delta = datetime.datetime.now() - start_time
	seconds = time_delta.seconds if time_delta.seconds > 0 else 1

	time_per_transaction = seconds / accuracy_results["total_processed"]
	transactions_per_minute = (accuracy_results["total_processed"] / seconds) * 60

	print("\nSPEED TESTS:")
	print("{0:35} = {1:11}".format("Total Time Taken", str(time_delta)[0:11]))
	print("{0:35} = {1:11.2f}".format("Time per Transaction (in seconds)", time_per_transaction))
	print("{0:35} = {1:11.2f}".format("Transactions Per Minute", transactions_per_minute))

	return {'time_delta': time_delta,
			'time_per_transaction': time_per_transaction,
			'transactions_per_minute': transactions_per_minute}

def all_CNN_accuracy():
	"""Run merchant CNN on a directory of Merchant Samples"""

	# Load Classifiers
	BANK_CNN = get_cnn("bank_merchant")
	CARD_CNN = get_cnn("card_merchant")

	# Connect to S3
	with nostdout():
		conn = get_s3_connection()

	bucket = conn.get_bucket("yodleemisc", Location.USWest2)

	# Test Bank CNN
	process_file_collection(bucket, "/vumashankar/CNN/bank/", BANK_CNN, "meerkat/classification/label_maps/reverse_bank_label_map.json")

	# Test Card CNN
	process_file_collection(bucket, "/vumashankar/CNN/card/", CARD_CNN, "meerkat/classification/label_maps/reverse_card_label_map.json")

def process_file_collection(bucket, prefix, classifier, classifier_id_map):
	"""Test a list of files"""

	label_map = load_params("meerkat/classification/label_maps/deep_clean_map.json")
	params = {}
	params["label_key"] = "MERCHANT_NAME"
	results = open("data/output/per_merchant_tests_" + prefix.split('/')[-2] + ".csv", "a")
	writer = csv.writer(results, delimiter=',', quotechar='"')
	writer.writerow(["Merchant", "Percent_Labeled", "Predictive_Accuracy"])

	for label_num in label_map.keys():

		merchant_name = label_map.get(label_num, "not_found")
		sample = bucket.get_key(prefix + label_num + ".txt.gz")
		if sample is None:
			continue
		file_name = "data/input/" + os.path.basename(sample.key)
		sample.get_contents_to_filename(file_name)

		df = pd.read_csv(file_name, na_filter=False, compression="gzip", quoting=csv.QUOTE_NONE, encoding="utf-8", sep='|', error_bad_lines=False)
		df.rename(columns={"DESCRIPTION": "DESCRIPTION_UNMASKED"}, inplace=True)
		df["MERCHANT_NAME"] = merchant_name
		df["GOOD_DESCRIPTION"] = merchant_name
		unzipped_file_name = "data/misc/Merchant Samples/" + label_num + ".txt"
		df.to_csv(unzipped_file_name, sep="|", mode="w", encoding="utf-8", index=False, index_label=False)
		safely_remove_file(file_name)

		params["verification_source"] = unzipped_file_name
		print("Testing Merchant: " + merchant_name)
		accuracy_results = CNN_accuracy(unzipped_file_name, classifier, classifier_id_map, label_map)
		print_results(accuracy_results)
		writer.writerow([merchant_name, accuracy_results["percent_labeled"], accuracy_results["predictive_accuracy"]])
		safely_remove_file(unzipped_file_name)

	results.close()

def run_from_command_line():
	"""Runs these commands if the module is invoked from the command line"""

	#print_results(vest_accuracy(params=None))
	all_CNN_accuracy()

if __name__ == "__main__":
	run_from_command_line()
