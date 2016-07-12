#!/usr/local/bin/python3.3
# pylint: disable=too-many-locals
# pylint: disable=pointless-string-statement

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

import os
import sys
import shutil
import logging
import argparse
import numpy as np

from .verify_data import verify_data
from .tools import (pull_from_s3, unzip_and_merge, seperate_debit_credit,
	extract_tarball, make_tarfile)
from meerkat.various_tools import push_file_to_s3

def parse_arguments(args):
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
	parser.add_argument("--train_size", help="Training data proportion, \
		default is 0.9.", default=0.9, type=float)
	parser.add_argument("-d", "--debug", help="log at DEBUG level",
		action="store_true")
	parser.add_argument("-v", "--info", help="log at INFO level",
		action="store_true")

	args = parser.parse_args(args)

	if args.model_type == 'subtype' and args.credit_or_debit == '':
		raise Exception('For subtype data you need to declare debit or credit.')

	if args.debug:
		logging.basicConfig(level=logging.DEBUG)
	elif args.info:
		logging.basicConfig(level=logging.INFO)
	return args

def random_split(df, train_size):
	"""Randomly partition df into train_df and test_df"""
	msk = np.random.rand(len(df)) <= train_size
	return {"train" : df[msk], "test" : df[~msk]}

def make_save_function(col, dirt):
	"""Decorate save_result"""
	def save_result(results, train_or_test):
		"""Save results and name it with train_or_test"""
		kwargs = {"cols" : col, "index" : False, 'sep' : '|'}
		path = dirt + train_or_test + '.csv'
		results[train_or_test].to_csv(path, **kwargs)
	return save_result

def main_split_data(args):
	"""Loads, splits and uploads data according to args"""
	model_type = args.model_type
	bank_or_card = args.bank_or_card
	data_type = model_type + "/" + bank_or_card
	credit_or_debit = args.credit_or_debit
	if model_type != "merchant":
		data_type = data_type + '/' + credit_or_debit

	bucket = "s3yodlee" if args.bucket == '' else args.bucket
	default_prefix = 'meerkat/cnn/data/'
	prefix = default_prefix + data_type + '/' if args.input_dir == '' else args.input_dir

	file_name = "input.tar.gz" if args.file_name == '' else args.file_name
	extension = ".tar.gz"
	output_file = "preprocessed.tar.gz"

	version = prefix[prefix.rfind("/", 0, len(prefix) - 1)+1:len(prefix)-1]
	save_path = './data/input/' + data_type + '_' + version +'/'
	save_path_input = save_path + 'input/'
	os.makedirs(save_path_input, exist_ok=True)
	save_path_preprocessed = save_path + 'preprocessed/'
	os.makedirs(save_path_preprocessed, exist_ok=True)

	input_file = pull_from_s3(bucket=bucket, prefix=prefix, extension=extension,
		file_name=file_name, save_path=save_path_input)

	if model_type == 'merchant':
		df, input_json_file = unzip_and_merge(input_file, bank_or_card)

		logging.info('Validating {0} {1} data'.format(model_type, bank_or_card))
		verify_data(csv_input=df, json_input=input_json_file,
			cnn_type=[model_type, bank_or_card])

	else:
		extract_tarball(input_file, save_path_input)
		input_csv_file = ""
		input_json_file = ""
		for file_name in os.listdir(save_path_input):
			if file_name.endswith(".json"):
				input_json_file = save_path_input + file_name
			if file_name.endswith(".csv"):
				input_csv_file = save_path_input + file_name

		df = seperate_debit_credit(input_csv_file, credit_or_debit, model_type)

		logging.info('Validating {0} {1} {2} data.'.\
			format(model_type, bank_or_card, credit_or_debit))
		verify_data(csv_input=df, json_input=input_json_file,
			cnn_type=[model_type, bank_or_card, credit_or_debit])

	# Save Results
	save = make_save_function(df.columns, save_path_preprocessed)
	results = random_split(df, args.train_size)
	save(results, 'test')
	save(results, 'train')

	del df
	del results

	label_map_path = save_path_preprocessed + "label_map.json"
	os.rename(input_json_file, label_map_path)
	stats_path = "data/CNN_stats"
	if not os.path.exists(stats_path):
		os.makedirs(stats_path)
	logging.info("Using shutil to copy label_map.json")
	shutil.copyfile(label_map_path, stats_path + "/" + "label_map.json")
	logging.info("label_map.json moved")
	make_tarfile(output_file, save_path_preprocessed)
	push_file_to_s3(output_file, bucket, prefix)

	shutil.rmtree(save_path_input)
	logging.info("remove directory of input files at: {0}".format(save_path_input))
	os.remove("./preprocessed.tar.gz")
	if model_type == "merchant":
		merchant_unzip_path = "./merchant_" + bank_or_card + "_unzip/"
		shutil.rmtree(merchant_unzip_path)
		logging.info("remove directory of unzipped merchant data at: {0}".format(merchant_unzip_path))

	logging.info('{0} uploaded to {1}'.format(output_file, bucket + '/' + prefix))

if __name__ == "__main__":
	ARGS = parse_arguments(sys.argv[1:])
	main_split_data(ARGS)
