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
python3 -m meerkat.tools.subtype_process \
data/input/Bank_complete_data_subtype_original.csv credit card
"""
#####################################################

import logging
import os
import sys
from .tools import (load, get_label_map, get_test_and_train_dataframes,
	get_json_and_csv_files)

# Load Data
if __name__ == "__main__":
	input_file, debit_or_credit, bank_or_card = sys.argv[1:]
	df, class_names = load(input_file=input_file, debit_or_credit=debit_or_credit)
	# Create a label map
	label_map = get_label_map(df=df, class_names=class_names)
	# Reverse the mapping (label_number: class_name)
	label_map = dict(zip(label_map.values(), label_map.keys()))
	# Make Test and Train
	results = get_test_and_train_dataframes(df=df)
	#Save
	output_path = "/data/preprocessed/"
	# Create an output directory if it does not exist
	os.makedirs(output_path, exist_ok=True)
	file_names = get_json_and_csv_files(output_path=output_path,
		debit_or_credit=debit_or_credit, bank_or_card=bank_or_card,
		label_map=label_map,
		df_test=results["df_test"],
		df_rich_train=results["df_rich_train"],
		df_poor_test=results["df_poor_test"],
		df_poor_train=results["df_poor_train"])
	logging.info("File names are {0}".format(file_names))

