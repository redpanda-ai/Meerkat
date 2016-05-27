#/usr/local/bin/python3.3
# pylint: disable=pointless-string-statement
"""This module loads and evaluates a trained CNN on a provided 
test set. It produces various stats and a confusion matrix for analysis

@author: Oscar Pan
"""

#################### USAGE ##########################
"""
python3 -m meerkat.classification.classification_report \
<path_to_classifier> \
<path_to_testdata> \
<path_to_label_map> \
<ground_truth_label_key> \
--model_name <optional_tensor_name> \
--doc_key <optional_primary_doc_key> \
--secdoc_key <optional_secondary_doc_key> \
--machine_label_key <optional_predicted_label_key>
# If working with merchant data
# If only want to return confusion matrix and performance statistics
--fast_mode

# Key values will be shifted to upper case.
"""
#####################################################

################### REFERENCE #######################
"""
An entry of machine_labeled has such format:
{'AMOUNT': 9.84,
 'DESCRIPTION': ' CKCD DEBIT 03/30 SICILIA PIZZA',
 'DESCRIPTION_UNMASKED': ' CKCD DEBIT 03/30 SICILIA PIZZA',
 'LEDGER_ENTRY': 'debit',
 'PREDICTED_SUBTYPE': 'Purchase - Purchase',
 'PROPOSED_SUBTYPE': 'Purchase - Purchase',
 'TRANSACTION_DATE': '2013-12-30',
 'UNIQUE_TRANSACTION_ID': 19}
"""
#####################################################

import argparse
import csv
import logging
import os
import sys
import pandas as pd

from meerkat.classification.load_model import get_tf_cnn_by_path
from meerkat.various_tools import load_params, load_piped_dataframe
from meerkat.classification.tools import reverse_map

def parse_arguments(args):
	""" Create the parser """
	parser = argparse.ArgumentParser(description="Test a CNN against a dataset and\
		return performance statistics")
	# Required arguments
	parser.add_argument('model', help='Path to the model under test')
	parser.add_argument('data', help='Path to the test data')
	parser.add_argument('label_map', help='Path to a label map')
	parser.add_argument('human_label_key', type=lambda x: x.upper(),
		help="Header name of the ground truth label column")
	# Optional arguments
	parser.add_argument('--model_name', default='', help='Name of the tensor stored in graph.')
	parser.add_argument('--doc_key', type=lambda x: x.upper(), default='DESCRIPTION_UNMASKED',
		help='Header name of primary transaction description column')
	parser.add_argument('--secdoc_key', default='DESCRIPTION', type=lambda x: x.upper(),
		help='Header name of secondary transaction description in case primary is empty')
	parser.add_argument('--machine_label_key', type=lambda x: x.upper(), default='PREDICTED_CLASS',
		help='Header name of predicted class column')
	parser.add_argument('--fast_mode', action='store_true', help='Use fast mode to save i/o time.')
	parser.add_argument("-d", "--debug", help="Show 'debug'+ level logs", action="store_true")
	parser.add_argument("-v", "--info", help="Show 'info'+ level logs", action="store_true")
	return parser.parse_args(args)

def compare_label(*args, **kwargs):
	"""similar to generic_test in accuracy.py, with unnecessary items dropped"""
	machine, cnn_column, human_column, conf_mat = args[:]
	doc_key = kwargs.get("doc_key")
	correct = []
	mislabeled = []

	# Test Each Machine Labeled Row
	for machine_row in machine:
		# Update conf_mat
		column = int(machine_row['PREDICTED_INDEX']) - 1
		row = int(machine_row['ACTUAL_INDEX']) - 1
		conf_mat[row][column] += 1

		# If fast mode True then do not record
		if not kwargs.get('fast_mode', False):
			# Predicted label matches human label
			if machine_row[cnn_column] == machine_row[human_column]:
				correct.append([machine_row[doc_key], machine_row[human_column]])
			else:
				mislabeled.append([machine_row[doc_key], machine_row[human_column],
					machine_row[cnn_column]])
	return mislabeled, correct, conf_mat


def get_write_func(filename, header):
	"""Have a write function"""
	file_exists = False
	def write_func(data):
		"""Have a write function"""
		if len(data) > 0:
			logging.info("Saving transactions to {0}".format(filename))
			nonlocal file_exists
			mode = "a" if file_exists else "w"
			add_head = False if file_exists else header
			df = pd.DataFrame(data)
			df.to_csv(filename, mode=mode, index=False, header=add_head,
				sep='|')
			file_exists = True
		else:
			#It's important to write empty files too
			logging.debug("Writing empty file {0}".format(filename))
			open(filename, 'a').close()
	return write_func

def count_transactions(csv_file):
	"""count number of transactions in csv_file"""
	with open(csv_file) as temp:
		reader = csv.reader(temp, delimiter='|')
		_ = reader.__next__()
		return sum([1 for i in reader])

def get_calculations(df, rows, label_map):
	"""Runs the formula for several valuable metrics."""
	#Convert to 0-indexed confusion matrix
	df.rename(columns=lambda x: int(x) - 1, inplace=True)
	#First order calculations
	true_positive = pd.DataFrame(df.iat[i, i] for i in range(rows))
	false_positive = pd.DataFrame(pd.DataFrame(df.sum(axis=0)).values - true_positive.values,
		columns=true_positive.columns)
	false_negative = pd.DataFrame(pd.DataFrame(df.sum(axis=1)).values - true_positive.values,
		columns=true_positive.columns)
	true_negative = pd.DataFrame(
		[df.drop(i, axis=1).drop(i, axis=0).sum().sum() for i in range(rows)])
	#Second order calculations
	accuracy = true_positive.sum() / df.sum().sum()
	precision = true_positive / (true_positive + false_positive)
	recall = true_positive / (true_positive + false_negative)
	specificity = true_negative / (true_negative + false_positive)
	#Third order calculation
	f_measure = 2 * precision * recall / (precision + recall)
	#Write out the classification report
	label = pd.DataFrame(label_map, index=[0]).transpose()
	label.index = label.index.astype(int)
	label = label.sort_index()
	num_labels = len(label_map)
	label.index = range(num_labels)

	return [("Accuracy", accuracy), ("Class", label), ("True Positive", true_positive),
		("False Positive", false_positive), ("False Negative", false_negative),
		("True Negative", true_negative), ("Precision", precision),
		("Recall", recall), ("Specificity", specificity), ("F Measure", f_measure)]

def get_classification_report(confusion_matrix_file, label_map):
	"""Produce a classification report for a particular confusion matrix"""
	df = pd.read_csv(confusion_matrix_file)
	rows, cols = df.shape
	if rows != cols:
		logging.critical("Rows: {0}, Columns {1}".format(rows, cols))
		logging.critical("Unable to make a square confusion matrix, aborting.")
		raise Exception("Unable to make a square confusion matrix, aborting.")
	else:
		logging.debug("Confusion matrix is a proper square, continuing")

	calculations = get_calculations(df, rows, label_map)

	#Create a classification report
	feature_list, feature_labels = [y for _, y in calculations], [x for x, _ in calculations]
	#Craft the report
	classification_report = pd.concat(feature_list, axis=1)
	classification_report.columns = feature_labels
	#Setting rows to be 1-indexed
	classification_report.index = range(1, rows + 1)

	logging.debug("Classification Report:\n{0}".format(classification_report))
	logging.info("Accuracy is: {0}".format(classification_report.iloc[0]["Accuracy"]))
	report_path = 'data/CNN_stats/classification_report.csv'
	logging.info("Classification Report saved to: {0}".format(report_path))
	classification_report.to_csv(report_path, index=False)

def test_classifier(args, my_classifier):
	"""Loop of doom"""
	chunk_count = 0
	processed = 0.0
	reader = load_piped_dataframe(args.data, chunksize=1000)
	fill_description = lambda x: x[args.sec_doc_key] if x[args.doc_key] == '' else x[args.doc_key]
	for chunk in reader:
		processed += len(chunk)
		my_progress = str(round(((processed / my_classifier.total_transactions) * 100), 2)) + '%'
		logging.info("Evaluating {0} of the testset".format(my_progress))
		logging.warning("Testing chunk {0}.".format(chunk_count))
		if args.sec_doc_key != '':
			chunk[args.doc_key] = chunk.apply(fill_description, axis=1)
		transactions = chunk.to_dict('records')
		machine_labeled = my_classifier.classifier(transactions, doc_key=args.doc_key,
			label_key=args.machine_label_key)

		# Add indexes for labels
		for item in machine_labeled:
			item['ACTUAL_INDEX'] = my_classifier.reversed_label_map[item[args.human_label_key]]
			item['PREDICTED_INDEX'] = my_classifier.reversed_label_map[item[args.machine_label_key]]

		mislabeled, correct, my_classifier.confusion_matrix =\
			compare_label(machine_labeled, args.machine_label_key, args.human_label_key,
			my_classifier.confusion_matrix, doc_key=args.doc_key, fast_mode=args.fast_mode)

		# Save
		my_classifier.write_mislabeled(mislabeled)
		my_classifier.write_correct(correct)
		chunk_count += 1
	#We now have an object containing our test results, which we return
	return my_classifier

class Classifier(object):
	"""This class contains members useful for evaluating the accuracy of a classifier"""
	def __init__(self, args, path):
		"""This function initializes a Classifier object"""
		#We start with a blank confusion matrix
		self.label_map = load_params(args.label_map)
		get_key = lambda x: x['label'] if isinstance(x, dict) else x
		self.label_map = dict(zip(self.label_map.keys(), map(get_key, self.label_map.values())))
		num_labels = len(self.label_map)
		self.confusion_matrix = [
			[0 for i in range(num_labels + 1)] for j in range(num_labels)]
		self.reversed_label_map = reverse_map(self.label_map)
		model_name = False if args.model_name == '' else args.model_name
		self.classifier = get_tf_cnn_by_path(args.model, args.label_map, model_name=model_name),
		self.write_mislabeled = get_write_func(path + "mislabeled.csv",
			['TRANSACTION_DESCRIPTION', 'ACTUAL', 'PREDICTED'])
		self.write_correct = get_write_func(path + "correct.csv",
			['TRANSACTION_DESCRIPTION', 'ACTUAL'])
		self.total_transactions = count_transactions(args.data)

def get_tested_classifier(args):
	"""This function returns a fully populated confusion matrix and label map"""

	# Ensure a path to save data locally
	path = 'data/CNN_stats/'
	os.makedirs(path, exist_ok=True)
	#Create a blank Classifier object to hold relevant members like the confusion_matrix
	#and label_map
	my_classifier = Classifier(args, path)
	logging.info("Total number of transactions: {0}".format(my_classifier.total_transactions))
	logging.info("Testing begins.")
	return test_classifier(args, my_classifier)

# Main
def main_process(args):
	"""This is the main stream"""
	tested_classifier = get_tested_classifier(args)

	#Make a square confusion matrix dataframe, df
	df = pd.DataFrame(tested_classifier.confusion_matrix)
	df = df.drop(df.columns[[-1]], axis=1)
	#Make sure the confusion matrix is a square
	rows, cols = df.shape
	if rows != cols:
		logging.critical("Rows: {0}, Columns {1}".format(rows, cols))
		logging.critical("Unable to make a square confusion matrix, aborting.")
		raise Exception("Unable to make a square confusion matrix, aborting.")
	else:
		logging.debug("Confusion matrix is a proper square, continuing")
	#Make sure the confusion matrix is 1-indexed, to match the label_map
	df.rename(columns=lambda x: int(x) + 1, inplace=True)
	df.index = range(1, rows + 1)
	#Save the confusion matrix out to a file
	confusion_matrix_path = 'data/CNN_stats/confusion_matrix.csv'
	rows, cols = df.shape
	logging.debug("Rows: {0}, Columns {1}".format(rows, cols))
	df.to_csv('data/CNN_stats/confusion_matrix.csv', index=False)
	logging.info("Confusion matrix saved to: {0}".format(confusion_matrix_path))
	get_classification_report(confusion_matrix_path, tested_classifier.label_map)

if __name__ == "__main__":
	ARGS = parse_arguments(sys.argv[1:])
	LOG_FORMAT = "%(asctime)s %(levelname)s: %(message)s"
	if ARGS.debug:
		logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)
	elif ARGS.info:
		logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
	else:
		logging.basicConfig(format=LOG_FORMAT, level=logging.WARNING)
	main_process(ARGS)
