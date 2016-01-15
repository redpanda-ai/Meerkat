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


import pandas as pd
import sys
import csv

from meerkat.classification.lua_bridge import get_cnn_by_path


default_doc_key = 'DESCRIPTION_UNMASKED'
default_machine_label_key = 'PREDICTED_SUBTYPE'
default_human_label_key = 'PROPOSED_SUBTYPE'

def compare_label(*args, **kwargs):
	"""similar to generic_test in accuracy.py, with unnecessary items dropped"""
	machine, human, cnn_column, human_column = args[:]
	doc_key = kwargs.get("doc_key", default_doc_key)

	# Create Quicker Lookup
	index_lookup = {row["UNIQUE_TRANSACTION_ID"]: row for row in human}

	unpredicted = []
	needs_hand_labeling = []
	correct = []
	mislabeled = []
	# Test Each Machine Labeled Row
	for _, machine_row in enumerate(machine):

		#Get human answer index
		key = machine_row["UNIQUE_TRANSACTION_ID"]
		human_row = index_lookup.get(key)

		# Continue if Unlabeled
		if machine_row[cnn_column] == "":
			unpredicted.append([machine_row[doc_key], human_row[human_column]])
			continue

		# Identify unlabeled points
		if not human_row or not human_row.get(human_column):
			needs_hand_labeling.append(machine_row[doc_key])
			continue

		# predicted label matches human label
		if machine_row[cnn_column] == human_row[human_column]:
			correct.append([human_row[doc_key], human_row[human_column]])
			continue

		mislabeled.append([human_row[doc_key], human_row[human_column],
			machine_row[cnn_column]])

	return mislabeled, correct, unpredicted, needs_hand_labeling

def make_unique_ID(row):
	"""assign each transaction a unique ID"""
	if row['UNIQUE_TRANSACTION_ID'] == '0':
		return row['TRANSACTION_UNMASKED']+ ' ' + str(row['AMOUNT'])
	else:
		return row['UNIQUE_TRANSACTION_ID']


# Main
classifier = get_cnn_by_path(sys.argv[1], sys.argv[2])
test_file = sys.argv[3]
reader = pd.read_csv(test_file, chunksize=1000, na_filter=False,
	quoting=csv.QUOTE_NONE, encoding='utf-8', sep='|', error_bad_lines=False)

for chunk in reader:
	chunk['UNIQUE_TRANSACTION_ID'] = chunk.apply(make_unique_ID, axis=1)
	transactions = chunk.to_dict('records')
	machine_labeled = classifier(transactions, doc_key=default_doc_key,
		label_key=default_machine_label_key)
	mislabeled, correct, unpredicted, needs_hand_labeling = compare_label(
		machine_labeled, transactions, default_machine_label_key,
		default_human_label_key, doc_key=default_doc_key)

	# Save
	if len(mislabeled) > 0:
		df = pd.DataFrame(mislabeled)
		df.to_csv('data/CNN_stats/mislabeled.csv', mode='a', index=False,
			header=['TRANSACTION_DESCRIPTION', 'ACTUAL', 'PREDICTED'])

	if len(correct) > 0:
		df = pd.DataFrame(correct)
		df.to_csv('data/CNN_stats/correct.csv', mode='a', index=False,
			header=['TRANSACTION_DESCRIPTION', 'ACTUAL'])

	if len(unpredicted) > 0:
		df = pd.DataFrame(unpredicted)
		df.to_csv('data/CNN_stats/unpredicted.csv', mode='a', index=False,
			header=['TRASACTION_DESCRIPTION', 'ACTUAL'])

	if len(needs_hand_labeling) > 0:
		df = pd.DataFrame(needs_hand_labeling)
		df.to_csv('data/CNN_stats/need_labeling.csv', mode='a', index=False,
			header=['TRANSACTION_DESCRIPTION'])

