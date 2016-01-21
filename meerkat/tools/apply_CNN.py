#!/usr/local/bin/python3.3

"""This utility loads a training CNN (sequentail_*.t7b) and a test set,
predicts labels of the test set. It also out perfomance statistics.

@author: Oscar Pan
"""

#################### USAGE ##########################
"""
python3 -m meerkat.tools.apply_CNN  <path_to_classifier> \
 <path_to_classifier_output_map> <path_to_test_data>
"""
#####################################################

################### REFERENCE #######################
"""
An entry of machine_labeled has such format:
{'AMOUNT': 9.84,
 'DESCRIPTION': ' CKCD DEBIT 03/30 SICILIA PIZZA',
 'DESCRIPTION_UNMASKED': ' CKCD DEBIT 03/30 SICILIA PIZZA',
 'LABEL': 27,
 'LEDGER_ENTRY': 'debit',
 'PREDICTED_SUBTYPE': 'purchase - purchase',
 'PROPOSED_SUBTYPE': 'purchase - purchase',
 'TRANSACTION_DATE': '2013-12-30',
 'UNIQUE_TRANSACTION_ID': 19}
"""
#####################################################

import pandas as pd
import sys
import csv
import json
import os
import numpy as np

from meerkat.classification.lua_bridge import get_cnn_by_path


default_doc_key = 'DESCRIPTION_UNMASKED'
default_machine_label_key = 'PREDICTED_SUBTYPE'
default_human_label_key = 'PROPOSED_SUBTYPE'

def compare_label(*args, **kwargs):
	"""similar to generic_test in accuracy.py, with unnecessary items dropped"""
	machine, cnn_column, human_column, cm = args[:]
	doc_key = kwargs.get("doc_key", default_doc_key)

	unpredicted = []
	needs_hand_labeling = []
	correct = []
	mislabeled = []

	# Test Each Machine Labeled Row
	for machine_row in machine:

		# Update cm
		# predicted_label is None if a predicted subtype is ""
		if machine_row['PREDICTED_LABEL'] is None:
			column = num_labels
		else:
			column = machine_row['PREDICTED_LABEL'] - 1
		row = machine_row['LABEL'] - 1
		cm[row][column] += 1

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

def load_and_reverse_label_map(filename):
	"""Load label map into a dict and switch keys and values"""
	input_file = open(filename, encoding='utf-8')
	label_map = json.load(input_file)
	reversed_map = dict((value, int(key)) for key, value in label_map.items())
	input_file.close()
	return reversed_map

def fill_description(df):
	"""Replace Description_Unmasked"""
	if df['DESCRIPTION_UNMASKED'] == "":
		return df['DESCRIPTION']
	else:
		return df['DESCRIPTION_UNMASKED']

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
classifier = get_cnn_by_path(sys.argv[1], sys.argv[2])
reader = pd.read_csv(sys.argv[3], chunksize=1000, na_filter=False,
	quoting=csv.QUOTE_NONE, encoding='utf-8', sep='|', error_bad_lines=False)
reversed_label_map = load_and_reverse_label_map(sys.argv[2])
num_labels = len(reversed_label_map)
confusion_matrix = [[0 for i in range(num_labels + 1)] for j in range(num_labels)]

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
	chunk[default_doc_key] = chunk.apply(fill_description, axis=1)
	transactions = chunk.to_dict('records')
	machine_labeled = classifier(transactions, doc_key=default_doc_key,
		label_key=default_machine_label_key)

	# Add indexes for predicted labels
	for item in machine_labeled:
		if item[default_machine_label_key] == "":
			item['PREDICTED_LABEL'] = None
			continue
		item['PREDICTED_LABEL'] = reversed_label_map[item[default_machine_label_key]]

	mislabeled, correct, unpredicted, needs_hand_labeling, confusion_matrix =\
		compare_label(machine_labeled, default_machine_label_key,
		default_human_label_key, confusion_matrix, doc_key=default_doc_key)

	# Save
	write_mislabeled(mislabeled)
	write_correct(correct)
	write_unpredicted(unpredicted)
	write_needs_hand_labeling(needs_hand_labeling)

# calculate recall, precision, false +/-, true +/- from confusion maxtrix
true_positive = pd.DataFrame([confusion_matrix[i][i] for i in range(num_labels)])
cm = pd.DataFrame(confusion_matrix)
actual = pd.DataFrame(cm.sum(axis=1))
recall = true_positive / actual
#if we use pandas 0.17 we can do the rounding neater
recall = np.round(recall, decimals=4)
column_sum = pd.DataFrame(cm.sum()).ix[:,:num_labels]
false_positive = column_sum - true_positive
precision = true_positive / column_sum
precision = np.round(precision, decimals=4)
false_negative = actual - true_positive
label = pd.DataFrame(pd.read_json(sys.argv[2], typ='series')).sort_index()
label.index = range(num_labels)

stat = pd.concat([label, actual, true_positive, false_positive, recall, precision,
	false_negative], axis=1)
stat.columns = ['Class', 'Actual', 'True_Positive', 'False_Positive', 'Recall',
	'Precision', 'False_Negative(false_negative & unpredicted)']

cm = pd.concat([label, cm], axis=1)
cm.columns = ['Class'] + [str(x) for x in range(num_labels)] + ['Unpredicted']

stat.to_csv('data/CNN_stats/CNN_stat.csv', index=False)
cm.to_csv('data/CNN_stats/Con_Matrix.csv')

