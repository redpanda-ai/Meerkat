#!/usr/local/bin/python3.3

"""This module processes the subtype training data into a format
acceptable for usage with Crepe

Created on Nov 9, 2015
@author: @author: Matthew Sevrens
"""

import pandas as pd
import numpy as np

import json
import csv
import sys

#################### USAGE ##########################

# python3 -m meerkat.tools.process_subtype_data [container]
# python3 -m meerkat.tools.process_subtype_data card

#####################################################

def load_label_map(filename):
	"""Load a permanent label map"""
	input_file = open(filename, encoding='utf-8')
	label_map = json.loads(input_file.read())
	input_file.close()
	return label_map

def process_data(groups, ledger_entry):
	"""Process and save data"""

	label_map = load_label_map("meerkat/classification/label_maps/" + sys.argv[1] + "_" + ledger_entry + "_subtype_label_map.json")
	label_map = dict(zip(label_map.values(), label_map.keys()))
	df = groups[ledger_entry]
	processed = []

	# Process Rows
	for index, row in df.iterrows():

		try:
			masked_row = {'CLASS_NUM' : label_map[row["PROPOSED_SUBTYPE"]], 
				  	      'BLANK': "",
				          'DESCRIPTION': ' '.join(row["DESCRIPTION"].split())
			}
			processed.append(masked_row)
		except: 
			print("The type " + row["PROPOSED_SUBTYPE"] + " is not allowed for the " + sys.argv[1] + " " + ledger_entry + " model")
			continue

		if row["DESCRIPTION_UNMASKED"] != "":
			unmasked_row = {'CLASS_NUM' : label_map[row["PROPOSED_SUBTYPE"]], 
			  	            'BLANK': "",
			                'DESCRIPTION': ' '.join(row["DESCRIPTION_UNMASKED"].split())
			}
			processed.append(unmasked_row)

	# Convert to Dataframe and Shuffle
	training_data_df = pd.DataFrame(processed)
	training_data_df = training_data_df.reindex(np.random.permutation(training_data_df.index))

	# Make Test and Train
	msk = np.random.rand(len(training_data_df)) < 0.90
	train = training_data_df[msk]
	test = training_data_df[~msk]

	# Save
	file_prefix = sys.argv[1] + "_" + ledger_entry
	train.to_csv("data/output/" + file_prefix + "_train_subtype.csv", cols=["CLASS_NUM", "BLANK", "DESCRIPTION"], header=False, index=False, index_label=False)
	test.to_csv("data/output/" + file_prefix + "_test_subtype.csv", cols=["CLASS_NUM", "BLANK", "DESCRIPTION"], header=False, index=False, index_label=False)

df = pd.read_csv("data/input/aggregated_" + sys.argv[1] + "_subtype_training_data.csv", na_filter=False, quoting=csv.QUOTE_NONE, encoding="utf-8", sep='|', error_bad_lines=False)
grouped = df.groupby('LEDGER_ENTRY', as_index=False)
groups = dict(list(grouped))

process_data(groups, "credit")
process_data(groups, "debit")