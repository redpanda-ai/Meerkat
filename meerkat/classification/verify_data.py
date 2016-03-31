#/usr/local/bin/python3.3

"""This module verify csv and json:

1 csv
1.1 csv data should have at lease 99% transactions as the original data set
1.2 verify csv data format is correct for the type of CNN being trained
1.3 verify any class should have at least 500 transactions

2 json
2.1 verify json format is correct
2.2 verify json is 1-indexed
2.3 verify no duplicate label numbers in json
2.4 verify no duplicate label names in json

3 consistency between csv and json
3.1 verify no missing label numbers in json
3.2 verify no extra label numbers in json
3.3 verify no missing label numbers in json
3.4 verify no extra label names in json

@author: Tina Wu
"""

#################### USAGE ##########################
"""
verify_data.py [-h] [--credit_or_debit CREDIT_OR_DEBIT]
                      csv_input json_input merchant_or_subtype bank_or_card

positional arguments:
  csv_input             what is the csv data, allowed format: a directory path
                        containing all csv files; a csv file path; pandas data
                        frame
  json_input            where is the json file
  merchant_or_subtype   What kind of dataset do you want to process, subtype
                        or merchant
  bank_or_card          Whether we are processing card or bank transactions

optional arguments:
  -h, --help            show this help message and exit
  --credit_or_debit CREDIT_OR_DEBIT
                        What kind of transactions do you wanna process, debit
                        or credit
"""
#####################################################

import os
import json
import csv
import logging
import sys
import collections
import argparse
import pandas as pd

WARNING_THRESHOLD = 0.01
CRITICAL_THRESHOLD = 0.05

def add_err_msg(label_csv, label_json, numbers_or_names):
	"""Generate error message"""
	err_msg = ""
	missing_list = sorted(list(set(label_csv) - set(label_json)))
	missing_str = ", ".join(("\"" + str(item) + "\"") for item in missing_list)
	if missing_str != "":
		err_msg += "There are missing class " + numbers_or_names + " in json: " + missing_str + "\n"
	extra_list = sorted(list(set(label_json) - set(label_csv)))
	extra_str = ", ".join(("\"" + str(item) + "\"") for item in extra_list)
	if extra_str != "":
		err_msg += "There are extra class " + numbers_or_names + " in json: " + extra_str + "\n"
	return err_msg

def check_json_and_csv_consistency(label_names_csv, label_names_json, label_numbers_json):
	"""Check consistency between csv data and json data"""
	label_numbers_csv = list(range(1, (len(label_names_csv) + 1)))

	err_msg = ""
	# Verify that there is no missing or extra class number in json
	if label_numbers_json != label_numbers_csv:
		err_msg += add_err_msg(label_numbers_csv, label_numbers_json, "numbers")

	# Verify that there is no missing or extra class name in json
	if label_names_json != label_names_csv:
		err_msg += add_err_msg(label_names_csv, label_names_json, "names")

	if err_msg != "":
		logging.critical("There are inconsistency errors between csv and json:\n{0}".format(err_msg))
		sys.exit()
	else:
		logging.info("json is consistent with csv\n")

def dict_raise_on_duplicates(ordered_pairs):
	"""Verify that there is no duplicate key in json"""
	dictionary = {}
	for key, value in ordered_pairs:
		if key in dictionary:
			raise ValueError("duplicate key: %r" % (key,))
		else:
			dictionary[key] = value
	return dictionary

def load_json(json_input):
	"""Verify that json can be loaded and there is no duplicate keys in json"""
	try:
		json_file = open(json_input, encoding='utf-8')
		try:
			label_map_json = json.load(json_file, object_pairs_hook=dict_raise_on_duplicates)
			logging.info("json file format is correct")
			return label_map_json
		except ValueError as err:
			logging.critical("json file is mal-formatted: {0}".format(err))
			sys.exit()
		json_file.close()
	except IOError:
		logging.critical("json file not found")
		sys.exit()

def parse_arguments():
	"""This function parses arguments from our command line."""
	parser = argparse.ArgumentParser()

	# Required arugments
	parser.add_argument("csv_input", help="what is the csv data, allowed format: a directory path \
		containing all csv files; a csv file path; pandas data frame")
	parser.add_argument("json_input", help="where is the json file")
	parser.add_argument("merchant_or_subtype",
		help="What kind of dataset do you want to process, subtype or merchant")
	parser.add_argument("bank_or_card", help="Whether we are processing card or \
		bank transactions")

	# Optional arguments
	parser.add_argument("--credit_or_debit", default='',
		help="What kind of transactions do you wanna process, debit or credit")

	args = parser.parse_args()
	if args.merchant_or_subtype == 'subtype' and args.credit_or_debit == '':
		raise Exception('For subtype data you need to declare debit or credit')
	return args

def read_csv_to_df(csv_input, cnn_type):
	"""Read csv file into pandas data frames"""
	df = []
	if os.path.isdir(csv_input):
		samples = []
		for i in os.listdir(csv_input):
			if i.endswith(".csv"):
				samples.append(i)

		for sample in samples:
			df_one_sample = pd.read_csv(csv_input + "/" + sample, na_filter=False, encoding="utf-8",
				sep="|", error_bad_lines=False, quoting=csv.QUOTE_NONE)
			df.append(df_one_sample)
		merged = pd.concat(df, ignore_index=True)
		return merged

	else:
		df = pd.read_csv(csv_input, quoting=csv.QUOTE_NONE, na_filter=False,
			encoding="utf-8", sep='|', error_bad_lines=False, low_memory=False)
		df['LEDGER_ENTRY'] = df['LEDGER_ENTRY'].str.lower()
		grouped = df.groupby('LEDGER_ENTRY', as_index=False)
		groups = dict(list(grouped))
		df = groups[cnn_type[2]]
		return df

def verify_csv(**kwargs):
	"""Verify csv data"""
	csv_input = kwargs["csv_input"]
	cnn_type = kwargs["cnn_type"]

	df = []
	if isinstance(csv_input, str):
		df = read_csv_to_df(csv_input, cnn_type)
	elif isinstance(csv_input, pd.core.frame.DataFrame):
		df = csv_input
	else:
		logging.critical("csv_input should be a string path or pandas dataframe")
		sys.exit()

	label_names_csv, label_counts_csv = verify_total_numbers(df, cnn_type)
	verify_csv_format(df, cnn_type)
	verify_numbers_in_each_class(label_names_csv, label_counts_csv)

	logging.info("csv is verified\n")
	return label_names_csv

def verify_csv_format(df, cnn_type):
	"""Verify csv data format is correct for the type of CNN being trained"""
	column_header = list(df.columns.values)
	column_header.sort()
	
	merchant_header = ['DESCRIPTION', 'DESCRIPTION_UNMASKED', 'MERCHANT_NAME']
	subtype_header = ['AMOUNT', 'DESCRIPTION', 'DESCRIPTION_UNMASKED', 'LEDGER_ENTRY',
		'PROPOSED_SUBTYPE', 'TRANSACTION_DATE', 'UNIQUE_TRANSACTION_ID']

	cnn_column_header = merchant_header if cnn_type[0] == "merchant" else subtype_header

	if sorted(column_header) != sorted(cnn_column_header):
		logging.critical("csv data format is incorrect")
		sys.exit()
	logging.info("csv data format is correct")

def verify_data(**kwargs):
	"""This function verifies csv data and json data"""
	logging.basicConfig(level=logging.INFO)

	csv_input = kwargs["csv_input"]
	json_input = kwargs["json_input"]
	cnn_type = kwargs["cnn_type"]

	label_names_csv = verify_csv(csv_input=csv_input, cnn_type=cnn_type)
	label_names_json, label_numbers_json = verify_json(json_input=json_input)
	check_json_and_csv_consistency(label_names_csv, label_names_json, label_numbers_json)

	logging.info("json and csv validation success")

def verify_json(**kwargs):
	"""verify json label map"""
	json_input = kwargs["json_input"]

	label_map_json = load_json(json_input)

	# Create a sorted list for label numbers in json
	keys_json = [int(x) for x in label_map_json.keys()]
	label_numbers_json = sorted(list(keys_json))

	verify_json_1_indexed(label_numbers_json)

	# Create a sorted list for label names in json
	label_names_json = []
	for value in label_map_json.values():
		label_names_json.append(value["label"])
	label_names_json = sorted(label_names_json)

	verify_json_no_dup_names(label_names_json)

	logging.info("json is verified\n")
	return label_names_json, label_numbers_json

def verify_json_1_indexed(label_numbers_json):
	"""Verify that the json map is 1-indexed"""
	if 0 in label_numbers_json:
		logging.critical("json is 0-indexed")
		sys.exit()
	logging.info("json is 1-indexed")

def verify_json_no_dup_names(label_names_json):
	"""Verify that there is no duplicate class name in json"""
	unique_label_names_json = set(label_names_json)
	if len(label_names_json) != len(unique_label_names_json):
		counter_names = collections.Counter(label_names_json)
		duplicate_names_list = []
		for name in counter_names:
			if counter_names[name] > 1:
				duplicate_names_list.append(name)
		duplicate_names = ", ".join(item for item in set(duplicate_names_list))
		logging.critical("There are duplicate class names in json: {0}".format(duplicate_names))
		sys.exit()
	logging.info("There is no duplicate class name in json")

def verify_numbers_in_each_class(label_names_csv, label_counts_csv):
	"""Verify that for any particular class, there are at least 500 transactions"""
	err_msg = ""
	for i in range(len(label_names_csv)):
		if label_counts_csv[i] < 500:
			err_msg += "{:<40}".format(label_names_csv[i]) + "{:<25}".format(str(label_counts_csv[i])) + "\n"
	if err_msg != "":
		err_msg = ("{:<40}".format("Class Name") + "{:<25}".format("Number of Transactions") +
			"\n") + err_msg
		logging.critical("The following classes have less than 500 transactions:\n{0} ".format(err_msg))
		sys.exit()
	else:
		logging.info("For any particular class, there are at least 500 transactions")

def verify_total_numbers(df, cnn_type):
	"""Check that in csv there should be enough transactions"""
	# Data sets should have at least 99% transactions as the original data sets
	original_data_sizes = {
		# Original merchant data size resource:
		# Issue 511 - Update Merchant Models for Card and Bank
		"merchant_bank": 23942324,
		"merchant_card": 16228034,
		# Original bank data size resource:
		# s3://s3yodlee/development/bank/aggregated_bank_subtype_training_data.csv
		"subtype_bank_debit": 117773,
		"subtype_bank_credit": 29654,
		# Original card data size resource:
		# s3://s3yodlee/development/card/aggregated_card_subtype_training_data.csv
		"subtype_card_debit": 151336,
		"subtype_card_credit": 23442
	}

	cnn_str = "_".join(item for item in cnn_type)
	original_data_size = original_data_sizes[cnn_str]

	err_msg = ""
	# get the total percent of transactions less than original data set
	total_percent = (original_data_size - len(df)) / original_data_size
	if total_percent >= WARNING_THRESHOLD:
		err_msg += ("Data set size of csv is " + "{:.1%}".format(total_percent) +
			" smaller than original data set size\n")
		err_msg += ("{:<40}".format("Data set size of csv: ") +
			"{:>15,}".format(len(df)) + "\n")
		err_msg += ("{:<40}".format("Original data set size: ") +
			"{:>15,}".format(original_data_size) + "\n")
	else:
		logging.info("Data set size of csv is verified: {0:>15,}".format(len(df)))

	# Generate count numbers for labels in csv
	label_key_csv = "MERCHANT_NAME" if cnn_type[0] == "merchant" else "PROPOSED_SUBTYPE"
	label_names_csv = sorted(df[label_key_csv].value_counts().index.tolist())
	label_counts_csv = df[label_key_csv].value_counts()

	# For merchant CNN, null class should have at least 99% transactions
	# as null class in original data sets
	if cnn_type[0] == "merchant":
		null_class_size = label_counts_csv[""]
		original_null_class_sizes = {
			# Original merchant null class size resource:
			# Issue 511 - Update Merchant Models for Card and Bank
			"bank": 12425494,
			"card": 4193517
		}
		original_null_class_size = original_null_class_sizes[cnn_type[1]]
		null_percent = (original_null_class_size - null_class_size) / original_null_class_size
		if null_percent >= WARNING_THRESHOLD:
			err_msg += ("Null class size in csv is " + "{:.1%}".format(null_percent) +
				" smaller than null class size in original data set\n")
			err_msg += "{:<40}".format("Null class size in csv: ") + "{:>15,}".format(null_class_size) + "\n"
			err_msg += ("{:<40}".format("Null class size in original data set: ") +
				"{:>15,}".format(original_null_class_size))
		else:
			logging.info("Null class size is verified:      {0:>15,}".format(null_class_size))

	if err_msg != "":
		if total_percent >= CRITICAL_THRESHOLD or null_percent >= CRITICAL_THRESHOLD:
			logging.critical("{0}".format(err_msg))
			sys.exit()
		else:
			logging.warning("{0}".format(err_msg))

	return label_names_csv, label_counts_csv

if __name__ == "__main__":
	args = parse_arguments()
	cnn_type = [args.merchant_or_subtype, args.bank_or_card]
	if args.credit_or_debit != "":
		cnn_type.append(args.credit_or_debit)

	verify_data(csv_input=args.csv_input, json_input=args.json_input, cnn_type=cnn_type)
