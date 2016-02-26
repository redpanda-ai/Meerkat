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
<required 'merchant' or 'subtype'> \
<required 'bank' or 'card'> \
<optional bucket name> \
<optional input_directory> \
<optional raw_data_file_name> \
<optional training_data_proportion> (0 to 1)

python3 -m meerkat.classification.split_data subtype bank \
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
from .tools import pull_from_s3, unzip_and_merge, seperate_debit_credit
from plumbum import local

def parse_arguments():
	"""This function parses arguments from our command line."""
	parser = argparse.ArgumentParser("split_data")
	# Required arguments
	parser.add_argument("merchant_or_subtype", help="What kind of dataset \
		do you want to split, merchant or subtype?")
	parser.add_argument("bank_or_card", help="Whether we are processing \
		card or bank transactions.")

	#optional arguments
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
	if args.debug:
		logging.basicConfig(level=logging.DEBUG)
	elif args.info:
		logging.basicConfig(level=logging.INFO)
	return args

def random_split(df, train_size):
	msk = np.random.rand(len(df)) <= train_size
	return {"train" : df[msk], "test" : df[~msk]}

def make_save_function(col, dirt, merchant_or_subtype, bank_or_card, date):
	def save_result(results, train_or_test, credit_or_debit=''):
		credit_or_debit = '_'*(credit_or_debit!='') + credit_or_debit
		kwargs = {"cols" : col, "index" : False, 'sep' : '|'}
		path = dirt + merchant_or_subtype + '_' + bank_or_card +\
			credit_or_debit + '_' +\
			train_or_test + '_' + date + '.csv'
		results[train_or_test].to_csv(path, **kwargs)
	return save_result

def main_split_data():
	args = parse_arguments()
	merchant_or_subtype = args.merchant_or_subtype
	bank_or_card = args.bank_or_card
	# To pull data according to bank or card (subtype has distinct dirs, 
	# merchant has bank and card under the same directory)
	data_type = merchant_or_subtype + '_' + bank_or_card
	default_dir_paths = {'merchant_card' : "cadusumi/tde_merchants_phase2/",
		'merchant_bank' : "cadusumi/tde_merchants_phase2/",
		'subtype_card' :"hvudumala/Type_Subtype_finaldata/Card/",
		'subtype_bank' : "hvudumala/Type_Subtype_finaldata/Bank/"
		}
	default_buckets = {'merchant' : "yodleemisc",
		'subtype' : "yodleemisc"
		}
	default_file_types = {'merchant' : '.tar.gz', 'subtype' : '.csv'}

	if args.bucket == '':
		bucket = default_buckets[merchant_or_subtype]
	else:
		bucket = args.bucket

	if args.input_dir == '':
		prefix = default_dir_paths[data_type]
	else:
		prefix = args.input_dir

	extension = default_file_types[merchant_or_subtype]
	save_path = './'

	input_file = pull_from_s3(bucket=bucket, prefix=prefix, extension=extension,
		file_name=args.file_name, save_path=save_path)


	dir_paths = {'subtype_card_debit': 's3://s3yodlee/data/subtype/card/debit',
		'subtype_card_credit': 's3://s3yodlee/data/subtype/card/credit/',
		'subtype_bank_debit': 's3://s3yodlee/data/subtype/bank/debit/',
		'subtype_bank_credit': 's3://s3yodlee/data/subtype/bank/credit/',
		'merchant_bank': 's3://s3yodlee/data/merchant/bank/',
		'merchant_card': 's3://s3yodlee/data/merchant/card/'
		}

	date = datetime.now()
	date = str(date.month).zfill(2) + str(date.day).zfill(2) + str(date.year)
	if merchant_or_subtype == 'merchant':
		dirt = save_path + data_type + '/'
		os.makedirs(dirt, exist_ok=True)
		df, label_map_path = unzip_and_merge(input_file, bank_or_card)
		save = make_save_function(df.columns, dirt, merchant_or_subtype,
			bank_or_card, date)
		# validate_data(df, label_map_path, blablabal)
		results = random_split(df, args.train_size)
		save(results, 'train')
		save(results, 'test')
		local['mv'][label_map_path][dirt]()
		local['aws']['s3']['sync'][dirt][dir_paths[data_type]]()
	else:
		df_credit, df_debit = seperate_debit_credit(input_file)
		# validate_data()
		# save debit
		dirt_debit = save_path + data_type + '_debit/'
		os.makedirs(dirt_debit, exist_ok=True)
		results_debit = random_split(df_debit, args.train_size)
		save = make_save_function(df_debit.columns, dirt_debit,
			merchant_or_subtype, bank_or_card, date)
		save(results_debit, 'train', credit_or_debit='debit')
		save(results_debit, 'test', credit_or_debit='debit')
		local['aws']['s3']['sync'][dirt_debit][dir_paths\
			[data_type + '_debit']]()
		# save credit
		dirt_credit = save_path + data_type + '_credit/'
		os.makedirs(dirt_credit, exist_ok=True)
		results_credit = random_split(df_credit, args.train_size)
		save = make_save_function(df_credit.columns, dirt_credit,
			merchant_or_subtype, bank_or_card, date)
		save(results_credit, 'train', credit_or_debit='credit')
		save(results_credit, 'test', credit_or_debit='credit')
		local['aws']['s3']['sync'][dirt_credit][dir_paths\
			[data_type + '_credit']]()

if __name__ == "__main__":
	main_split_data()
