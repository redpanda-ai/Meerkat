#!/usr/local/bin/python3.3

"""This script streamlines an entire CNN training process from pulling raw
data from s3 to applying the trained CNN to a test set and return
performane matrics.
@author: J. Andrew Key
@author: Jie Zhang
@author: Oscar Pan
"""

############################## USAGE ############################
"""
python3 -m meerkat.classification.stream \
<required 'merchant' or 'subtype'> \
<required 'bank' or 'card'> \
<required training_file_name> \
<required test_file_name> \
<required 'debit' or 'credit' for 'subtype'> \
<optional name_of_ouput_directory> \
<optional bucket name> \
<optional input_directory>
<optional raw_train_file_name> \
<optional raw_test_file_name> \
<optional debug_flag> \
<optional log_flag>

python3 -m meerkat.classification.stream merchant card \
merchant_card_train_02262016.csv merchant_card_test_02262016.csv
--output_dir output_CNN
"""
################################################################
import argparse
import logging
import os

from meerkat.classification.preprocess import preprocess
from meerkat.classification.automate import main_stream as check_accuracy
from meerkat.classification.tools import (cap_first_letter, pull_from_s3,
	convert_csv_to_torch_7_binaries, create_new_configuration_file,
	copy_file, execute_main_lua)
from meerkat.tools.CNN_stats import main_process as apply_cnn

def parse_arguments():
	"""This function parses arguments from our command line."""
	parser = argparse.ArgumentParser("stream")
	# Required arugments
	parser.add_argument("merchant_or_subtype",
		help="What kind of dataset do you want to process, subtype or merchant?")
	parser.add_argument("bank_or_card", help="Whether we are processing card or \
		bank transactions.")
	parser.add_argument("train_file", help="Name of training file to be pulled")
	parser.add_argument("test_file", help="Name of test file to be pulled")

	# Optional arguments
	parser.add_argument("--label_map", help="Name of label map be pulled")
	parser.add_argument("--output_dir", help="Where do you want to write out all \
		of your files? By default it will go to meerkat/data/", default='')
	parser.add_argument("--credit_or_debit", default='',
		help="What kind of transactions do you wanna process, debit or credit?")
	parser.add_argument("--bucket", help="Input bucket name, default is s3yodlee.",
		default = 's3yodlee')
	parser.add_argument("--input_dir", help="Path of the directory immediately\
		containing input file", default='')
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

def main_stream():
	"""It all happens here"""
	args = parse_arguments()
	bucket = args.bucket
	bank_or_card = args.bank_or_card
	credit_or_debit = args.credit_or_debit
	merchant_or_subtype = args.merchant_or_subtype
	data_type = merchant_or_subtype + '_' + bank_or_card + '_' +\
		credit_or_debit
	dir_paths = {'subtype_card_debit': 'data/subtype/card/debit/',
		'subtype_card_credit': 'data/subtype/card/credit/',
		'subtype_bank_debit': 'data/subtype/bank/debit/',
		'subtype_bank_credit': 'data/subtype/bank/credit/',
		'merchant_bank_': 'data/merchant/bank/',
		'merchant_card_': 'data/merchant/card/'
		}
	if args.input_dir == '':
		prefix = dir_paths[data_type]
	else:
		prefix = args.input_dir

	if args.output_dir == '':
		save_path = "./data/input/" + data_type + "/"
	else:
		save_path = args.output_dir + '/'*(args.output_dir[-1] != '/')
	os.makedirs(save_path, exist_ok=True)

	train_file = pull_from_s3(bucket=bucket, prefix=prefix, extension='.csv',
		file_name=args.train_file, save_path=save_path)
	test_file = pull_from_s3(bucket=bucket, prefix=prefix, extension='.csv',
		file_name=args.test_file, save_path=save_path)
	label_map = pull_from_s3(bucket=bucket, prefix=prefix, extension='.json',
		file_name=args.label_map, save_path=save_path)
	#2. Slice it into dataframes and make a mapping file.
	train_poor, val_poor, num_of_classes = preprocess(
		train_file, label_map, merchant_or_subtype, bank_or_card,
		credit_or_debit, output_path=save_path)
	#3.  Use qlua to convert the files into training and testing sets.
	train_poor = convert_csv_to_torch_7_binaries(train_poor)
	val_poor = convert_csv_to_torch_7_binaries(val_poor)
	#4 Create a new configuration file based on the number of classes.
	create_new_configuration_file(num_of_classes, save_path, train_poor, val_poor)
	#5 Copy main.lua and data.lua to output directory.
	copy_file("meerkat/classification/lua/main.lua", save_path)
	copy_file("meerkat/classification/lua/data.lua", save_path)
	copy_file("meerkat/classification/lua/model.lua", save_path)
	copy_file("meerkat/classification/lua/train.lua", save_path)
	copy_file("meerkat/classification/lua/test.lua", save_path)
	#6 Excuete main.lua and send to background.
	execute_main_lua(save_path, "main.lua")
	#7 Check training progress
	best_model_path = check_accuracy(save_path)
	print('The path to the best model is {0}.'.format(best_model_path))
	#8 apply final model to test set and save all metrics
	ground_truth_labels = {'merchant' : 'MERCHANT_NAME',
		'subtype' : 'PROPOSED_SUBTYPE'}
	args.model = best_model_path
	args.testdata = test_file
	args.label_map = label_map
	args.doc_key = 'DESCRIPTION_UNMASKED'
	args.secondary_doc_key = 'DESCRIPTION'
	args.label_key = ground_truth_labels[merchant_or_subtype]
	args.predicted_key = 'PREDICTED_CLASS'
	args.is_merchant = (merchant_or_subtype == 'merchant')
	args.fast_mode = True
	print('Apply the best CNN to test data and calculate performance metrics')
	apply_cnn(args)
	print('The whole streamline process has finished')

# The main program starts here if run from the command line.
if __name__ == "__main__":
	main_stream()

