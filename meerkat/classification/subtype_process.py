#!/usr/local/bin/python3.3

"""This script loads raw bank subtype data and partitions it into training
and test data. It also produces a index-label mapping json file.
@author: Matthew Sevrens
@author: Oscar Pan
@author: J. Andrew Key
"""

#################### USAGE ##########################
"""
python3 process.py [file_name] [ledger_entry] [bank_or_card]
python3 -m meerkat.classification.subtype_process \
data/input/Bank_complete_data_subtype_original.csv credit card
"""
#####################################################

import logging
import os
import sys
import pandas as pd
import csv
from .tools import (get_label_map, get_test_and_train_dataframes,
	cap_first_letter, get_json_and_csv_files, fill_description_unmasked)

def preprocess(input_file, debit_or_credit, bank_or_card,
		output_path='./data/preprocessed/'):
	logging.info("Loading subtype {0}  csv file ".format(debit_or_credit))
	df = pd.read_csv(filename, quoting=csv.QUOTE_NONE, na_filter=False,
		encoding="utf-8", sep='|', error_bad_lines=False, low_memory=False)
	class_names = df["PROPOSED_SUBTYPE"].value_counts().index.tolist()
	# Clean the "DESCRIPTION_UNMASKED" values within the dataframe
	df["DESCRIPTION_UNMASKED"] = df.apply(fill_description_unmasked, axis=1)
	# Create a label map
	label_map = get_label_map(class_names)
	# Map Numbers
	my_lambda = lambda x: label_map[x["PROPOSED_SUBTYPE"]]
	df['LABEL'] = df.apply(my_lambda, axis=1)

	# Reverse the mapping (label_number: class_name)
	reversed_label_map = dict(zip(label_map.values(), label_map.keys()))
	# Make Test and Train
	results = get_test_and_train_dataframes(df=df)
	# Create an output directory if it does not exist
	os.makedirs(output_path, exist_ok=True)
	file_names = get_json_and_csv_files(output_path=output_path,
		debit_or_credit=debit_or_credit, bank_or_card=bank_or_card,
		label_map=reversed_label_map,
		# df_test=results["df_rich_test"],
		# df_rich_train=results["df_rich_train"],
		df_poor_test=results["df_poor_test"],
		df_poor_train=results["df_poor_train"])
	logging.info("File names are {0}".format(file_names))
	return (file_names['train_poor'], file_names['test_poor'],
		len(class_names))

# Load Data
if __name__ == "__main__":
	input_file, debit_or_credit, bank_or_card = sys.argv[1:]
	_ = preprocess(input_file, debit_or_credit, bank_or_card)
