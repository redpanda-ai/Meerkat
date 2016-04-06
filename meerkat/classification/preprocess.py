#!/usr/local/bin/python3.3

"""This script loads raw data and label map and partitions it
into training and validation sets.
@author: Matthew Sevrens
@author: Oscar Pan
@author: J. Andrew Key
"""

#################### USAGE ##########################
"""
python3 process.py [file_name] [label_map] [merchant/subtype] [bank/card]

python3 -m meerkat.classification.preprocess \
data/input/Bank_complete_data_subtype_original.csv \
path_to_label_map merchant card -v
"""
#####################################################

import logging
import os
import sys
import pandas as pd
import csv
import json
import argparse

from meerkat.various_tools import load_piped_dataframe
from .tools import (get_label_map, get_test_and_train_dataframes,
	cap_first_letter, get_json_and_csv_files, fill_description_unmasked,
	get_csv_files, reverse_map)
from .verify_data import load_json

def parse_arguments():
	"""Parse arguments from command line."""
	parser = argparse.ArgumentParser("preprocess")
	# Required arguments
	parser.add_argument("file_name", help="Path of the input file.")
	parser.add_argument("label_map", help="Path of the label map.")
	parser.add_argument("merchant_or_subtype")
	parser.add_argument("bank_or_card")

	# Optional arguments
	parser.add_argument("--credit_or_debit",
		help="(This is required if data is subtype", default='')
	parser.add_argument("-d", "--debug", help="log at DEBUG level",
		action="store_true")
	parser.add_argument("-v", "--info", help="log at INFO level",
		action="store_true")

	args = parser.parse_args()
	if args.merchant_or_subtype == 'subtype' and args.credit_or_debit == '':
		raise Exception('For subtype data you need to declare debit or credit.')
	if args.debug:
		logging.basicConfig(level=logging.DEBUG)
	elif args.info:
		logging.basicConfig(level=logging.INFO)
	return args

def preprocess(input_file, label_map, merchant_or_subtype, bank_or_card,
		credit_or_debit, output_path='./data/preprocessed/'):
	logging.info("Loading {0} {1} csv file ".format(merchant_or_subtype,
		bank_or_card))
	df = load_piped_dataframe(input_file)
	# Clean the "DESCRIPTION_UNMASKED" values within the dataframe
	df["DESCRIPTION_UNMASKED"] = df.apply(fill_description_unmasked, axis=1)
	# Load label map
	# index-class label map
	label_map = load_json(label_map)
	# make a class-index label map
	reversed_label_map = reverse_map(label_map)
	ground_truth = {'subtype' : 'PROPOSED_SUBTYPE',
		'merchant' : 'MERCHANT_NAME'}
	label = ground_truth[merchant_or_subtype]
	if not len(label_map) == len(reversed_label_map) ==\
		len(df[label].value_counts()):
		raise Exception('Number of indexes does not match number of labels')
	# Map Numbers
	map_numbers = lambda x: reversed_label_map[x[label]]
	df['LABEL'] = df.apply(map_numbers, axis=1)

	# Make Test and Train
	results = get_test_and_train_dataframes(df)
	# Create an output directory if it does not exist
	os.makedirs(output_path, exist_ok=True)
	file_names = get_csv_files(output_path=output_path,
		credit_or_debit=credit_or_debit, bank_or_card=bank_or_card,
		label_map=reversed_label_map, merchant_or_subtype=merchant_or_subtype,
		# df_test=results["df_rich_test"],
		# df_rich_train=results["df_rich_train"],
		df_poor_test=results["df_poor_test"],
		df_poor_train=results["df_poor_train"])
	logging.info("File names are {0}".format(file_names))
	return (file_names['train_poor'], file_names['test_poor'], len(label_map))

# Load Data
if __name__ == "__main__":
	ARGS = parse_arguments()
	_ = preprocess(ARGS.file_name, ARGS.label_map, ARGS.merchant_or_subtype,
		ARGS.bank_or_card, ARGS.credit_or_debit)
