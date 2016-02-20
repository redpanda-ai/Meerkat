#/usr/local/bin/python3.3

"""This utility loads a trained CNN (sequentail_*.t7b) and a test set,
predicts labels of the test set. It also returns performance statistics.

@author: Oscar Pan
"""

#################### USAGE ##########################
"""
python3 -m meerkat.tools.apply_CNN \
-model <path_to_classifier> \
-data <path_to_testdata> \
-map <path_to_label_map> \
-label <ground_truth_label_key> \
-doc <optional_primary_doc_key> \
-secdoc <optional_secondary_doc_key> \
-predict <optional_predicted_label_key>
# If working with merchant data
-merchant
# If only want to return confusion matrix and performance statistics
-fast

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

import pandas as pd
import csv
import json
import os
import argparse
import numpy as np

from meerkat.classification.lua_bridge import get_cnn_by_path
from meerkat.various_tools import load_params

def get_parser():
	""" Create the parser """
	parser = argparse.ArgumentParser(description="Test a CNN against a dataset and\
		return performance statistics")
	parser.add_argument('--model', '-model', required=True,
		help='Path to the model under test')
	parser.add_argument('--testdata', '-data', required=True,
		help='Path to the test data')
	parser.add_argument('--label_map', '-map', required=True,
		help='Path to a label map')
	parser.add_argument('--doc_key', '-doc', required=False, type=lambda x: x.upper(),
		default='DESCRIPTION_UNMASKED',
		help='Header name of primary transaction description column')
	parser.add_argument('--secondary_doc_key', '-secdoc', required=False,
		default='DESCRIPTION', type=lambda x: x.upper(),
		help='Header name of secondary transaction description in case\
			primary is empty')
	parser.add_argument('--label_key', '-label', required=True,
		type=lambda x: x.upper(), help="Header name of the ground truth label column")
	parser.add_argument('--predicted_key', '-predict', required=False,
		type=lambda x: x.upper(), default='PREDICTED_CLASS',
		help='Header name of predicted class column')
	parser.add_argument('--is_merchant', '-merchant', required=False,
		action='store_true', help='If working on merchant data need to indicate True.')
	parser.add_argument('--fast_mode', '-fast', required=False, action='store_true',
		help='Use fast mode to save i/o time.')
	return parser

def run_from_command_line():
	"""Runs these commands if the module is invoked from the command line"""

	def compare_label(*args, **kwargs):
		"""similar to generic_test in accuracy.py, with unnecessary items dropped"""
		machine, cnn_column, human_column, cm = args[:]
		doc_key = kwargs.get("doc_key")

		unpredicted = []
		needs_hand_labeling = []
		correct = []
		mislabeled = []

		# Test Each Machine Labeled Row
		for machine_row in machine:

			# Update cm
			# predicted_label is None if a predicted subtype is ""
			if machine_row['ACTUAL_INDEX'] is None:
				pass
			elif machine_row['PREDICTED_INDEX'] is None:
				column = num_labels
				row = machine_row['ACTUAL_INDEX'] - 1
				cm[row][column] += 1
			else:
				column = machine_row['PREDICTED_INDEX'] - 1
				row = machine_row['ACTUAL_INDEX'] - 1
				cm[row][column] += 1

			# If fast mode True then do not record
			if not fast_mode:
				# Continue if unlabeled
				if machine_row[cnn_column] == "":
					unpredicted.append([machine_row[doc_key], machine_row[human_column]])
					continue

				# Identify unlabeled points
				if not machine_row[human_column]:
					needs_hand_labeling.append(machine_row[doc_key])
					continue

				# Predicted label matches human label
				if machine_row[cnn_column] == machine_row[human_column]:
					correct.append([machine_row[doc_key], machine_row[human_column]])
					continue

				mislabeled.append([machine_row[doc_key], machine_row[human_column],
					machine_row[cnn_column]])
		return mislabeled, correct, unpredicted, needs_hand_labeling, cm

	def fill_description(df):
		"""Replace Description_Unmasked"""
		if df[doc_key] == "":
			return df[sec_doc_key]
		return df[doc_key]

	def change_label_name(df):
		"""replace label '' with 'Null Class' """
		if df[human_label_key] == "":
			return "Null Class"
		return df[human_label_key]

	def get_write_func(filename, header):
		file_exists = False
		def write_func(data):
			if len(data) > 0:
				nonlocal file_exists
				mode = "a" if file_exists else "w"
				add_head = False if file_exists else header
				df = pd.DataFrame(data)
				df.to_csv(filename, mode=mode, index=False, header=add_head)
				file_exists = True
		return write_func

	# Main
	args = get_parser().parse_args()
	doc_key = args.doc_key
	sec_doc_key = args.secondary_doc_key
	machine_label_key = args.predicted_key
	human_label_key = args.label_key
	fast_mode = args.fast_mode
	reader = pd.read_csv(args.testdata, chunksize=1000, na_filter=False,
		quoting=csv.QUOTE_NONE, encoding='utf-8', sep='|', error_bad_lines=False)
	label_map = load_params(args.label_map)
	num_labels = len(label_map)
	class_names = list(label_map.values())

	# Create reversed label map and check it there are duplicate keys
	reversed_label_map = {}
	for key, value in label_map.items():
		if class_names.count(value) > 1:
			reversed_label_map[value] =\
				 sorted(reversed_label_map.get(value, []) + [int(key)])
		else:
			reversed_label_map[value] = int(key)

	if args.is_merchant:
		label_map["1"] = "Null Class"
		reversed_label_map["Null Class"] = reversed_label_map.pop("")

	confusion_matrix = [[0 for i in range(num_labels + 1)] for j in range(num_labels)]
	classifier = get_cnn_by_path(args.model, label_map)

	# Prepare for data saving
	path = 'data/CNN_stats/'
	os.makedirs(path, exist_ok=True)
	write_mislabeled = get_write_func(path + "mislabeled.csv",
		['TRANSACTION_DESCRIPTION', 'ACTUAL', 'PREDICTED'])
	write_correct = get_write_func(path + "correct.csv",
		['TRANSACTION_DESCRIPTION', 'ACTUAL'])
	write_unpredicted = get_write_func(path + "unpredicted.csv",
		["TRANSACTION_DESCRIPTION", 'ACTUAL'])
	write_needs_hand_labeling = get_write_func(path + "need_labeling.csv",
		["TRANSACTION_DESCRIPTION"])

	for chunk in reader:
		if sec_doc_key != '':
			chunk[doc_key] = chunk.apply(fill_description, axis=1)
		if args.is_merchant:
			chunk[human_label_key] = chunk.apply(change_label_name, axis=1)
		transactions = chunk.to_dict('records')
		machine_labeled = classifier(transactions, doc_key=doc_key,
			label_key=machine_label_key)

		# Add indexes for labels
		for item in machine_labeled:
			if item[human_label_key] == "":
				item['ACTUAL_INDEX'] = None
				continue
			temp = reversed_label_map[item[human_label_key]]
			if isinstance(temp, list):
				item['ACTUAL_INDEX'] = temp[0]
			else:
				item['ACTUAL_INDEX'] = temp
			if item[machine_label_key] == "":
				item['PREDICTED_INDEX'] = None
				continue
			temp = reversed_label_map[item[machine_label_key]]
			if isinstance(temp, list):
				item['PREDICTED_INDEX'] = temp[0]
			else:
				item['PREDICTED_INDEX'] = temp

		mislabeled, correct, unpredicted, needs_hand_labeling, confusion_matrix =\
			compare_label(machine_labeled, machine_label_key, human_label_key,
			confusion_matrix, doc_key=doc_key)

		# Save
		write_mislabeled(mislabeled)
		write_correct(correct)
		write_unpredicted(unpredicted)
		write_needs_hand_labeling(needs_hand_labeling)

	# Calculate recall, precision, false +/-, true +/- from confusion maxtrix
	true_positive = pd.DataFrame([confusion_matrix[i][i] for i in range(num_labels)])
	cm = pd.DataFrame(confusion_matrix)
	actual = pd.DataFrame(cm.sum(axis=1))
	recall = true_positive / actual
	# If we use pandas 0.17 we can do the rounding neater
	recall = np.round(recall, decimals=4)
	column_sum = pd.DataFrame(cm.sum()).ix[:,:num_labels]
	unpredicted = pd.DataFrame(cm.ix[:,num_labels])
	unpredicted.columns = [0]
	false_positive = column_sum - true_positive
	precision = true_positive / column_sum
	precision = np.round(precision, decimals=4)
	false_negative = actual - true_positive - unpredicted
	label = pd.DataFrame(pd.read_json(args.label_map, typ='series')).sort_index()
	label.index = range(num_labels)

	stat = pd.concat([label, actual, true_positive, false_positive, recall, precision,
		false_negative, unpredicted], axis=1)
	stat.columns = ['Class', 'Actual', 'True_Positive', 'False_Positive', 'Recall',
		'Precision', 'False_Negative', 'Unpredicted']

	cm = pd.concat([label, cm], axis=1)
	cm.columns = ['Class'] + [str(x) for x in range(1, num_labels + 1)] + ['Unpredicted']
	cm.index = range(1, num_labels + 1)

	stat.to_csv('data/CNN_stats/CNN_stat.csv', index=False)
	cm.to_csv('data/CNN_stats/Con_Matrix.csv')

if __name__ == "__main__":
	run_from_command_line()
