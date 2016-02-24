#/usr/local/bin/python3.3

"""This module load csv and json data and verify:

1 csv
1.1 csv dta should have same or more transactions as the original data set
1.2 verify csv data format is correct for the type of CNN being trained
1.3 verify any class has at least 500 transactions

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

To Do:
Integrate with stream.py, so that csv and json can be pulled from s3 automatedly

@author: Tina Wu
"""

#################### USAGE ##########################
"""
python3 -m meerkat.classification.verify_data \
<csv_input> <json_input> <cnn_type_arg1> <>cnn_type_arg2>

# Exampl - merchant_card CNN
python3 -m meerkat.classification.verify_data \
data/input/merchant_card data/input/merchant_card/card_merchant_label_map.json \
merchant card
# Note that: data/input/merchant_card is a directory containing all csv files 
# from s3

"""
#####################################################

import os
import json
import csv
import logging
import sys
import collections
import pandas as pd

def read_csv_to_df(csv_input, cnn_type):
	"""Read csv file into pandas data frames"""
	df = []
	if cnn_type[0] == "merchant":
		samples = []
		for i in os.listdir(csv_input):
			if i.endswith(".csv"):
				samples.append(i)

		for sample in samples:
			df_one_merchant = pd.read_csv(csv_input + "/" + sample, na_filter=False, encoding="utf-8",
				sep="|", error_bad_lines=False, quoting=csv.QUOTE_NONE, low_memory=False)
			df.append(df_one_merchant)
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

def verify_total_numbers(df, cnn_type):
	"""Check that in csv there should be enough transactions"""
	# Data sets should have at least 99% transactions as the original data sets
	original_data_sizes = {
		"merchant_bank": 23942324,
		"merchant_card": 16228034,
		"subtype_bank_debit": 117773,
		"subtype_bank_credit": 29654,
		"subtype_card_debit": 151336,
		"subtype_card_credit": 23442
	}

	cnn_str = "_".join(item for item in cnn_type)
	original_data_size = original_data_sizes[cnn_str]

	err_msg = ""
	total_percent = (original_data_size - len(df)) / original_data_size
	if total_percent >= 0.01:
		err_msg += ("Data set size of csv is " + "{:.1%}".format(total_percent) +
			" smaller than original data set size.\n")
		err_msg += "{:<40}".format("Data set size in csv: ") + str(len(df)) + "\n"
		err_msg += "{:<40}".format("Original data set size: ") + str(original_data_size) + "\n"

	# Generate count numbers for labels in csv
	label_key_csv = "MERCHANT_NAME"
	if cnn_type[0] == "subtype":
		label_key_csv = "PROPOSED_SUBTYPE"
	label_names_csv = sorted(df[label_key_csv].value_counts().index.tolist())
	label_counts_csv = df[label_key_csv].value_counts()

	# For merchant CNN, null class should have at least 99% transactions
	# as null class in original data sets
	if cnn_type[0] == "merchant":
		null_class_size = label_counts_csv[""]
		null_class_sizes = {
			"bank": 12425494,
			"card": 4193517
		}
		original_null_class_size = null_class_sizes[cnn_type[1]]
		null_percent = (original_null_class_size - null_class_size) / original_null_class_size
		if null_percent >= 0.01:
			err_msg += ("Null class size in csv is " + "{:.1%}".format(null_percent) +
				" smaller than null class size in original data set.\n")
			err_msg += "{:<40}".format("Null class size in csv: ") + str(null_class_size) + "\n"
			err_msg += ("{:<40}".format("Null class size in original data set: ") +
				str(original_null_class_size))

	if err_msg != "":
		if total_percent >= 0.05 or null_percent >= 0.05:
			logging.error("{0}".format(err_msg))
		else:
			logging.warning("{0}".format(err_msg))
		sys.exit()

	return label_names_csv, label_counts_csv

def verify_csv_format(df, cnn_type):
	"""Verify csv data format is correct for the type of CNN being trained"""
	column_header = list(df.columns.values)
	column_header.sort()
	
	merchant_column_header = ['AMOUNT', 'DESCRIPTION', 'DESCRIPTION_UNMASKED', 'GOOD_DESCRIPTION',
		'MERCHANT_NAME', 'TRANSACTION_DATE', 'TYPE', 'UNIQUE_MEM_ID', 'UNIQUE_TRANSACTION_ID']
	subtype_column_header = ['AMOUNT', 'DESCRIPTION', 'DESCRIPTION_UNMASKED', 'LEDGER_ENTRY',
		'PROPOSED_SUBTYPE', 'TRANSACTION_DATE', 'UNIQUE_TRANSACTION_ID']

	cnn_column_header = merchant_column_header
	if cnn_type[0] == "subtype":
		cnn_column_header = subtype_column_header

	if column_header != cnn_column_header:
		logging.error("csv data format is incorrect.")
		sys.exit()
	logging.info("csv data format is correct.")

def verify_numbers_in_each_class(label_names_csv, label_counts_csv):
	"""Verify that for any particular class, there're at least 500 transactions"""
	err_msg = ""
	for i in range(len(label_names_csv)):
		if label_counts_csv[i] < 500:
			err_msg += "{:<40}".format(label_names_csv[i]) + "{:<25}".format(str(label_counts_csv[i])) + "\n"
	if err_msg != "":
		err_msg = ("{:<40}".format("Class Name") + "{:<25}".format("Number of Transactions") +
			"\n") + err_msg
		logging.error("The following classes have less than 500 transactions:\n{0} ".format(err_msg))
		sys.exit()
	logging.info("For any particular class, there are at least 500 transactions")

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
		logging.error("csv_input should be a string path or pandas dataframe")
		sys.exit()

	label_names_csv, label_counts_csv = verify_total_numbers(df, cnn_type)
	verify_csv_format(df, cnn_type)
	verify_numbers_in_each_class(label_names_csv, label_counts_csv)

	return label_names_csv

def dict_raise_on_duplicates(ordered_pairs):
	"""Check duplicate keys in JSON"""
	dictionary = {}
	for key, value in ordered_pairs:
		if key in dictionary:
			raise ValueError("duplicate key: %r" % (key,))
		else:
			dictionary[key] = value
	return dictionary

def load_json(json_input):
	"""Verify JSON file is correct"""
	try:
		json_file = open(json_input, encoding='utf-8')
		try:
			label_map_json = json.load(json_file, object_pairs_hook=dict_raise_on_duplicates)
			return label_map_json
		except ValueError as err:
			logging.error("The label map json file is mal-formatted: {0}".format(err))
			sys.exit()
		json_file.close()
	except IOError:
		logging.error("Json file not found, aborting.")
		sys.exit()
	logging.info("JSON file format is correct")

def verify_json(**kwargs):
	"""verify json label map"""
	json_input = kwargs["json_input"]

	label_map_json = load_json(json_input)

	keys_json = [int(x) for x in label_map_json.keys()]
	label_numbers_json = sorted(list(keys_json))

	# Verify that the json map is 1-indexed
	if 0 in label_numbers_json:
		logging.error("JSON label map is 0-indexed")
		sys.exit()

	label_names_json = []
	for value in label_map_json.values():
		label_names_json.append(value["label"])
	label_names_json = sorted(label_names_json)

	# check there are no duplicate class names in json
	unique_label_names_json = set(label_names_json)
	if len(label_names_json) != len(unique_label_names_json):
		counter_names = collections.Counter(label_names_json)
		duplicate_names_list = []
		for name in counter_names:
			if counter_names[name] > 1:
				duplicate_names_list.append(name)
		duplicate_names = ', '.join(item for item in set(duplicate_names_list))
		logging.error("There are duplicate class names in json: {0}".format(duplicate_names))
		sys.exit()

	return label_names_json, label_numbers_json

def check_consistency(label_names_csv, label_names_json, label_numbers_json):
	"""Check consistency between csv data and json"""
	label_numbers_csv = list(range(1, (len(label_names_csv) + 1)))

	err_msg = ""
	# Verify that there are no missing or extra class numbers in json
	if label_numbers_json != label_numbers_csv:
		missing_numbers_list = sorted(list(set(label_numbers_csv) - set(label_numbers_json)))
		missing_numbers = ', '.join(str(item) for item in missing_numbers_list)
		if missing_numbers != "":
			err_msg += "There are missing class numbers in json: " + missing_numbers + "\n"

		extra_numbers_list = sorted(list(set(label_numbers_json) - set(label_numbers_csv)))
		extra_numbers = ', '.join(str(item) for item in extra_numbers_list)
		if extra_numbers != "":
			err_msg += "There are extra class numbers in json: " + extra_numbers + "\n"

	# Verify that there are no missing or extra class names in json
	if label_names_json != label_names_csv:
		missing_names_list = sorted(list(set(label_names_csv) - set(label_names_json)))
		missing_names = ', '.join(str(item) for item in missing_names_list)
		if missing_names != "":
			err_msg += "There are missing class names in json: " + missing_names + "\n"

		extra_names_list = sorted(list(set(label_names_json) - set(label_names_csv)))
		extra_names = ', '.join(str(item) for item in extra_names_list)
		if extra_names != "":
			err_msg += "There are extra class names in json: " + extra_names + "\n"

	if err_msg != "":
		logging.error("There are inconsistency errors between csv and json:\n{0}".format(err_msg))
		sys.exit()

def verify_data(**kwargs):
	"""This function verifies csv data and json label map"""
	logging.basicConfig(level=logging.INFO)

	csv_input = kwargs["csv_input"]
	json_input = kwargs["json_input"]
	cnn_type = kwargs["cnn_type"]

	label_names_csv = verify_csv(csv_input=csv_input, cnn_type=cnn_type)
	label_names_json, label_numbers_json = verify_json(json_input=json_input)
	check_consistency(label_names_csv, label_names_json, label_numbers_json)

	logging.info("CSV and JSON are verified")

if __name__ == "__main__":
	csv_input = sys.argv[1]
	json_input = sys.argv[2]
	cnn_type = sys.argv[3:]
	verify_data(csv_input=csv_input, json_input=json_input, cnn_type=cnn_type)
