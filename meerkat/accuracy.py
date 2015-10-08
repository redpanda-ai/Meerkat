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
import datetime
import logging
import os
import sys
import contextlib
import boto

from boto.s3.connection import Location

from itertools import zip_longest
import pandas as pd

from meerkat.various_tools import load_dict_list, progress, safely_remove_file
from meerkat.classification.lua_bridge import get_cnn, load_label_map

class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
	save_stdout = sys.stdout
	save_stderr = sys.stderr
	sys.stdout = DummyFile()
	sys.stderr = DummyFile()
	yield
	sys.stderr = save_stderr
	sys.stdout = save_stdout

def grouper(iterable):
	"""Returns batches of size 128 of iterable elements"""
	return zip_longest(*[iter(iterable)]*128,fillvalue={"DESCRIPTION_UNMASKED":""})
def get_s3_connection():
	"""Returns a connection to S3"""
	try:
		conn = boto.connect_s3()
	except boto.s3.connection.HostRequiredError:
		print("Error connecting to S3, check your credentials")
		sys.exit()
	return conn

def generic_test(machine, human, lists, column):
	"""Tests both the recall and precision of the pinpoint classifier against
	human-labeled training data."""

	sys.stdout.write('\n')
	doc_label = 'DESCRIPTION_UNMASKED'
	index_lookup = {}

	# Create Quicker Lookup
	for index, row in enumerate(human):
		key = str(row["UNIQUE_TRANSACTION_ID"])
		index_lookup[key] = index

	# Test Each Machine Labeled Row
	for index, machine_row in enumerate(machine):

		# Display progress
		progress(index, machine, message="complete with accuracy tests")

		# Continue if Unlabeled
		if machine_row[column] == "":
			lists["unlabeled"].append(machine_row[doc_label])
			continue

		# Verify Accuracy
		key = str(machine_row["UNIQUE_TRANSACTION_ID"])
		h_index = index_lookup.get(key, "")

		# Sort Into Lists
		if h_index == "":
			lists["needs_hand_labeling"].append(machine_row[doc_label])
			continue
		else: 
			human_row = human[h_index]
			if human_row[column] == "":
				lists["needs_hand_labeling"].append(machine_row[doc_label])
				continue
			elif machine_row[column] == human_row[column]:
				lists["correct"].append(human_row[doc_label] + \
				" (ACTUAL:" + human_row[column] + ")")
				continue
			else:
				lists["mislabeled"].append(human_row[doc_label] + " (ACTUAL: " \
 				+ human_row[column] + ")" + " (FOUND: " + machine_row[column] + ")")
				continue

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

def vest_accuracy(params, file_path=None, non_physical_trans=None,\
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
	verification_source = params.get("verification_source",\
		"data/misc/verifiedLabeledTrans.txt")
	human_labeled = load_dict_list(verification_source)

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

	# Test Classifier for recall and precision
	label_key = params.get("label_key", "FACTUAL_ID")
	generic_test(machine_labeled, human_labeled, my_lists, label_key)


	# Test Bulk (binary) Classifier for accuracy
	#test_bulk_classifier(human_labeled, non_physical_trans, my_lists)

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

def speed_vests(start_time, accuracy_results):
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


def apply_cnn(classifier, transactions):
	"""Apply the CNN to transactions"""

	batches = grouper(transactions)
	processed = []

	for batch in enumerate(batches):
		processed += classifier(batch, doc_key="DESCRIPTION_UNMASKED", \
label_key="MERCHANT_NAME")

	return processed[0:len(transactions)]

def per_merchant_accuracy(params, classifier):
	"""An easy way to test the accuracy of a small set
	provided a set of hyperparameters"""

	print("Testing sample: " + params["verification_source"])
	transactions = load_dict_list(params["verification_source"])
	labeled_trans = apply_cnn(classifier, transactions)
	accuracy_results = vest_accuracy(params, result_list=labeled_trans)
	print_results(accuracy_results)

	return accuracy_results

def CNN_accuracy():
	"""Run merchant CNN on a directory of Merchant Samples"""

	# Load Classifiers
	BANK_CNN = get_cnn("bank_merchant")
	CARD_CNN = get_cnn("card_merchant")

	# Connect to S3
	with nostdout():
		conn = get_s3_connection()

	bucket = conn.get_bucket("yodleemisc", Location.USWest2)

	# Test Bank CNN
	process_file_collection(bucket, "/vumashankar/CNN/bank/", BANK_CNN)

	# Test Card CNN
	process_file_collection(bucket, "/vumashankar/CNN/card/", CARD_CNN)

def process_file_collection(bucket, prefix, classifier):
	"""Test a list of files"""

	label_map = load_label_map\
	("meerkat/classification/label_maps/deep_clean_map.json")
	params = {}
	params["label_key"] = "MERCHANT_NAME"
	results = open\
	("data/output/per_merchant_tests_" + prefix.split('/')[-2] + ".csv", "a")
	writer = csv.writer(results, delimiter=',', quotechar='"')
	writer.writerow(["Merchant", "Recall", "Precision"])

	for label_num in label_map.keys():

		merchant_name = label_map.get(label_num, "not_found")
		sample = bucket.get_key(prefix + label_num + ".txt.gz")
		if sample == None: continue
		file_name = "data/input/" + os.path.basename(sample.key)
		sample.get_contents_to_filename(file_name)

		df = pd.read_csv(file_name, na_filter=False, compression="gzip",\
		 quoting=csv.QUOTE_NONE, encoding="utf-8", sep='|', error_bad_lines=False)
		df.rename(columns={"DESCRIPTION": "DESCRIPTION_UNMASKED"}, inplace=True)
		df["MERCHANT_NAME"] = merchant_name
		unzipped_file_name = "data/misc/Merchant Samples/" + label_num + ".txt"
		df.to_csv(unzipped_file_name, sep="|", mode="w", \
		encoding="utf-8", index=False, index_label=False)
		safely_remove_file(file_name)
		
		params["verification_source"] = unzipped_file_name
		print("Testing Merchant: " + merchant_name)
		accuracy_results = per_merchant_accuracy(params, classifier)
		writer.writerow([merchant_name, accuracy_results['total_recall'], \
		accuracy_results["precision"]])
		safely_remove_file(unzipped_file_name)

	results.close()

def print_results(results):
	"""Provide useful readable output"""

	if results is None:
		return

	sys.stdout.write('\n\n')

	print("STATS:")
	print("{0:35} = {1:11}".format("Total Transactions Processed",
		results['total_processed']))
	print("{0:35} = {1:10.2f}%".format("Total Labeled Physical",
		results['total_physical']))
	print("{0:35} = {1:10.2f}%".format("Total Labeled Non Physical",
		results['total_non_physical']))
	print("{0:35} = {1:10.2f}%".format("Binary Classifier Accuracy",
		results['binary_accuracy']))

	sys.stdout.write('\n')

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

	#print("", "UNLABELED:", '\n'.join(sorted(results['unlabeled'])), sep="\n")
	#print("", "MISLABELED:", '\n'.join(sorted(results['mislabeled'])), sep="\n")
	#print("", "MISLABELED BINARY:", '\n'.join(results['non_physical']),
	#	sep="\n")

def run_from_command_line(command_line_arguments):
	"""Runs these commands if the module is invoked from the command line"""
	
	#print_results(vest_accuracy(params=None))
	CNN_accuracy()

if __name__ == "__main__":
	run_from_command_line(sys.argv)
