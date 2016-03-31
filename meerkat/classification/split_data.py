#!/usr/local/bin/python3.3

"""This script downloads an entire raw data file from s3, performs data
validation to make sure future CNN training is possible. Then it splits
the data into training and test sets. Training set could be further splitted
into training and validation sets. The test set should only be exposed to
a trained classifier for performance analysis.
"""

########################## USAGE ###############################
"""
python3 -m meerkat.classification.split_data \
<required 'merchant' or 'subtype' or 'category'> \
<required 'bank' or 'card'> \
<optional credit_or_debit> \
<optional bucket_name> \
<optional input_directory> \
<optional raw_data_file_name> \
<optional training_data_proportion> (0 to 1)

python3 -m meerkat.classification.split_data subtype bank \
--credit_or_debit credit \
--train_size 0.88 \
--file_name Bank_complete_data_subtype_original_updated.csv
"""
################################################################

import logging
import argparse
import pandas as pd
import numpy as np
import os

from datetime import datetime
from .verify_data import verify_data
from .tools import pull_from_s3, unzip_and_merge, seperate_debit_credit
from plumbum import local

def parse_arguments():
	"""This function parses arguments from our command line."""
	parser = argparse.ArgumentParser("split_data")
	# Required arguments
	parser.add_argument("model_type", help="What kind of dataset \
		do you want to split, merchant or subtype?")
	parser.add_argument("bank_or_card", help="Whether we are processing \
		card or bank transactions.")

	#optional arguments
	parser.add_argument("--credit_or_debit", help="ledger entry", default='')
	parser.add_argument("--bucket", help="Input bucket name", default='')
	parser.add_argument("--input_dir", help="Path of the directory immediately \
		containing raw data file.", default='')
	parser.add_argument("--file_name", help="Name of file to be pulled.",
		default='')
	parser.add_argument("--train_size" , help="Training data proportion, \
		default is 0.9.", default=0.9, type=float)
	parser.add_argument("-d", "--debug", help="log at DEBUG level",
		action="store_true")
	parser.add_argument("-v", "--info", help="log at INFO level",
		action="store_true")

	args = parser.parse_args()

	if args.model_type == 'subtype' and args.credit_or_debit == '':
		raise Exception('For subtype data you need to declare debit or credit.')

	if args.debug:
		logging.basicConfig(level=logging.DEBUG)
	elif args.info:
		logging.basicConfig(level=logging.INFO)
	return args

def random_split(df, train_size):
	msk = np.random.rand(len(df)) <= train_size
	return {"train" : df[msk], "test" : df[~msk]}

def make_save_function(col, dirt):
	def save_result(results, train_or_test):
		kwargs = {"cols" : col, "index" : False, 'sep' : '|'}
		path = dirt + train_or_test + '.csv'
		results[train_or_test].to_csv(path, **kwargs)
	return save_result

def main_split_data(args):
	model_type = args.model_type
	bank_or_card = args.bank_or_card
	data_type = model_type + "_" + bank_or_card
	credit_or_debit = args.credit_or_debit
	if model_type != "merchant":
		data_type = data_type + '_' + credit_or_debit

	#TODO update default_dir_paths
	default_dir_paths = {
		'merchant_card' : "data/merchant/card/",
		'merchant_bank' : "data/merchant/bank/",
		'subtype_card_debit' : "data/subtype/card/debit/",
		'subtype_card_credit' : "data/subtype/card/credit/",
		'subtype_bank_debit' : "data/subtype/bank/debit",
		'subtype_bank_credit' : "data/subtype/bank/credit/"
	}

	bucket = "s3yodlee" if args.bucket == '' else args.bucket
	prefix = default_dir_paths[data_type] if args.input_dir == '' else args.input_dir
	file_name = "input.tar.gz" if args.file_name == '' else args.file_name
	extension = ".tar.gz"
	output_file = "output.tar.gz"
	dir_path = "s3://s3yodlee/" + prefix

	version = prefix[prefix.rfind("/", 0, len(prefix) - 1)+1:len(prefix)-1]
	save_path = './data/input/' + data_type + '_' + version +'/'
	save_path_input = save_path + 'input/'
	os.makedirs(save_path_input, exist_ok=True)
	save_path_output = save_path + 'output/'
	os.makedirs(save_path_output, exist_ok=True)

	input_file = pull_from_s3(bucket=bucket, prefix=prefix, extension=extension,
		file_name=file_name, save_path=save_path_input)

	if model_type == 'merchant':
		df, input_json_file = unzip_and_merge(input_file, bank_or_card)

		logging.info('Validating {0} {1} data'.format(model_type, bank_or_card))
		verify_data(csv_input=df, json_input=input_json_file,
			cnn_type=[model_type, bank_or_card])

	else:
		local['tar']['xf'][input_file]['-C'][save_path_input]()
		input_csv_file = ""
		input_json_file = ""
		for file_name in os.listdir(save_path_input):
			if file_name.endswith(".json"):
				input_json_file = save_path_input + file_name
			if file_name.endswith(".csv"):
				input_csv_file = save_path_input + file_name

		if credit_or_debit == "credit":
			df, _ = seperate_debit_credit(input_csv_file)
		else:
			_, df = seperate_debit_credit(input_csv_file)

		logging.info('Validating {0} {1} {2} data.'.\
			format(model_type, bank_or_card, credit_or_debit))
		verify_data(csv_input=df, json_input=input_json_file,
			cnn_type=[model_type, bank_or_card, credit_or_debit])

	save = make_save_function(df.columns, save_path_output)
	results = random_split(df, args.train_size)
	save(results, 'train')
	save(results, 'test')
	del df
	del results

	os.rename(input_json_file, save_path_output + "label_map.json")
	local['tar']['-zcvf'][output_file]['-C'][save_path_output]['.']()
	local['aws']['s3']['cp'][output_file][dir_path]()

	logging.info('{0} uploaded to {1}'.format(output_file, bucket + '/' + prefix))

if __name__ == "__main__":
	args = parse_arguments()
	main_split_data(args)
