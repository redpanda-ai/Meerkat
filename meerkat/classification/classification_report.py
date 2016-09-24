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
--predict_key <optional_predicted_label_key>
# If working with merchant data
# If only want to return confusion matrix and performance statistics
--fast_mode
# If model being tested is a SWS
--sws

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

from meerkat.classification.load_model import get_tf_cnn_by_path, get_sws_by_path
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
	parser.add_argument('label', help="Header name of the ground truth label column")
	# Optional arguments
	parser.add_argument("--sws", action="store_true",
		help="Use this argument if model being tested is sws"),
	parser.add_argument('--model_name', default='', help='Name of the tensor stored in graph.')
	parser.add_argument('--doc_key', type=lambda x: x.upper(), default='DESCRIPTION_UNMASKED',
		help='Header name of primary transaction description column')
	parser.add_argument('--secdoc_key', default='DESCRIPTION', type=lambda x: x.upper(),
		help='Header name of secondary transaction description in case primary is empty')
	parser.add_argument('--predict_key', type=lambda x: x.upper(), default='PREDICTED_CLASS',
		help='Header name of predicted class column')
	parser.add_argument('--fast_mode', action='store_true', help='Use fast mode to save i/o time.')
	parser.add_argument("-d", "--debug", help="Show 'debug'+ level logs", action="store_true")
	parser.add_argument("-v", "--info", help="Show 'info'+ level logs", action="store_true")
	return parser.parse_args(args)

def compare_label(*args, **kwargs):
	"""similar to generic_test in accuracy.py, with unnecessary items dropped"""
	machine, cnn_column, human_column, conf_mat, _ = args[:]
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

def get_classification_report(confusion_matrix_file, label_map, report_path):
	"""Produce a classification report for a particular confusion matrix"""
	df = pd.read_csv(confusion_matrix_file)
	rows, cols = df.shape
	if rows != cols:
		logging.critical("Rows: {0}, Columns {1}".format(rows, cols))
		logging.critical("Unable to make a square confusion matrix, aborting.")
		raise Exception("Unable to make a square confusion matrix, aborting.")
	else:
		logging.debug("Confusion matrix is a proper square, continuing")

	#Convert to 0-indexed confusion matrix
	df.columns = list(range(len(df)))
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

	#Create a classification report
	feature_list = [accuracy, label, true_positive, false_positive,
		false_negative, true_negative, precision, recall, specificity,
		f_measure]
	feature_labels = ["Accuracy", "Class", "True Positive", "False Positive",
		"False Negative", "True Negative", "Precision", "Recall", "Specificity",
		"F Measure"]
	#Craft the report
	classification_report = pd.concat(feature_list, axis=1)
	classification_report.columns = feature_labels
	#Setting rows to be 1-indexed
	classification_report.index = range(1, rows + 1)

	logging.debug("Classification Report:\n{0}".format(classification_report))
	logging.info("Accuracy is: {0}".format(classification_report.iloc[0]["Accuracy"]))
	logging.info("Classification Report saved to: {0}".format(report_path))
	classification_report.to_csv(report_path, index=False)

# Main
def main_process(args=None):
	"""This is the main stream"""
	if args is None:
		args = parse_arguments(sys.argv[1:])
	log_format = "%(asctime)s %(levelname)s: %(message)s"
	if args.debug:
		logging.basicConfig(format=log_format, level=logging.DEBUG)
	elif args.info:
		logging.basicConfig(format=log_format, level=logging.INFO)
	else:
		logging.basicConfig(format=log_format, level=logging.WARNING)
	sws = args.sws
	if sws:
		doc_key = "Description"
	else:
		doc_key = args.doc_key
	sec_doc_key = args.secdoc_key
	machine_label_key = args.predict_key
	human_label_key = args.label
	model_name = False if args.model_name == '' else args.model_name
	fast_mode = args.fast_mode
	reader = load_piped_dataframe(args.data, chunksize=1000)
	total_transactions = count_transactions(args.data)
	processed = 0.0
	label_map = load_params(args.label_map)

	get_key = lambda x: x['label'] if isinstance(x, dict) else x
	label_map = dict(zip(label_map.keys(), map(get_key, label_map.values())))
	reversed_label_map = reverse_map(label_map)
	num_labels = len(label_map)
	#class_names = list(label_map.values())

	confusion_matrix = [[0 for i in range(num_labels + 1)] for j in range(num_labels)]
	if sws:
		classifier = get_sws_by_path(args.model, args.label_map, model_name=model_name)
	else:
		classifier = get_tf_cnn_by_path(args.model, args.label_map, model_name=model_name)

	# Prepare for data saving
	path = 'data/CNN_stats/'
	os.makedirs(path, exist_ok=True)
	write_mislabeled = get_write_func(path + "mislabeled.csv",
		['TRANSACTION_DESCRIPTION', 'ACTUAL', 'PREDICTED'])
	write_correct = get_write_func(path + "correct.csv",
		['TRANSACTION_DESCRIPTION', 'ACTUAL'])

	fill_description = lambda x: x[sec_doc_key] if x[doc_key] == ''\
		else x[doc_key]
	chunk_count = 0

	logging.info("Total number of transactions: {0}".format(total_transactions))
	logging.info("Testing begins.")
	for chunk in reader:
		processed += len(chunk)
		my_progress = str(round(((processed/total_transactions) * 100), 2)) + '%'
		logging.info("Evaluating {0} of the testset".format(my_progress))
		logging.warning("Testing chunk {0}.".format(chunk_count))
		if sec_doc_key != '' and not sws:
			chunk[doc_key] = chunk.apply(fill_description, axis=1)
		if sws:
			translate_label = lambda x: "no" if x[human_label_key] == "" else "yes"
			chunk[human_label_key] = chunk.apply(translate_label, axis=1)
		transactions = chunk.to_dict('records')
		machine_labeled = classifier(transactions, doc_key=doc_key,
			label_key=machine_label_key)

		# Add indexes for labels
		for item in machine_labeled:
			item['ACTUAL_INDEX'] = reversed_label_map[item[human_label_key]]
			item['PREDICTED_INDEX'] = reversed_label_map[item[machine_label_key]]

		mislabeled, correct, confusion_matrix =\
			compare_label(machine_labeled, machine_label_key, human_label_key,
			confusion_matrix, num_labels, doc_key=doc_key, fast_mode=fast_mode)

		# Save
		write_mislabeled(mislabeled)
		write_correct(correct)

		chunk_count += 1
	#Make a square confusion matrix dataframe, df
	df = pd.DataFrame(confusion_matrix)
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
	report_path = 'data/CNN_stats/classification_report.csv'
	get_classification_report(confusion_matrix_path, label_map, report_path)


if __name__ == "__main__":
	main_process()
