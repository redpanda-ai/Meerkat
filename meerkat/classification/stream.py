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
<required name_of_ouput_directory> \
<required 'merchant' or 'subtype'> \
<required 'bank' or 'card'> \
<required 'debit' or 'credit' for 'subtype'> \
<optional bucket name> \
<optional input_directory>
<optional raw_data_file_name> \
<optional debug_flag> \
<optional log_flag>

python3 -m meerkat.classification.stream output_CNN subtype bank \
--debit_or_credit debit \
--bucket yodleemisc \
--input_dir  hvudumala/Type_Subtype_finaldata/Bank/ \
--file_name Bank_complete_data_subtype_original_updated.csv \
-d -v

"""
################################################################
import argparse
import logging

from .subtype_process import preprocess as subtype_preprocess
from .tools import (cap_first_letter, pull_from_s3,
	convert_csv_to_torch_7_binaries, create_new_configuration_file,
	copy_file, execute_main_lua)

def parse_arguments():
	"""This function parses arguments from our command line."""
	parser = argparse.ArgumentParser("stream")
	# Required arugments
	parser.add_argument("output_dir", help="Where do you want to write out all \
		of your files?")
	parser.add_argument("merchant_or_subtype",
		help="What kind of dataset do you want to process, subtype or merchant?")
	parser.add_argument("bank_or_card", help="Whether we are processing card or \
		bank transactions.")

	# Optional arguments
	parser.add_argument("--debit_or_credit", default='',
		help="What kind of transactions do you wanna process, debit or credit?")
	parser.add_argument("--bucket", help="Input bucket name, default is s3yodlee.",
		default = 's3yodlee')
	parser.add_argument("--input_dir", help="Path of the directory immediately\
		containing input file", default='')
	parser.add_argument("--file_name", help="Name of file to be pulled", default='')
	parser.add_argument("-d", "--debug", help="log at DEBUG level",
		action="store_true")
	parser.add_argument("-v", "--info", help="log at INFO level",
		action="store_true")
	args = parser.parse_args()
	if args.merchant_or_subtype == 'subtype' and args.debit_or_credit == '':
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
	debit_or_credit = args.debit_or_credit
	merchant_or_subtype = args.merchant_or_subtype
	data_type = merchant_or_subtype + '_' + bank_or_card + '_' + debit_or_credit
	dir_paths = {'subtype_card_debit': 'data/subtype/card/debit',
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

	extension, save_path = ".csv", "./data/input/" + merchant_or_subtype + "/"
	#1. Download data and save to input_path
	input_file = pull_from_s3(bucket=bucket, prefix=prefix, extension=extension,
		file_name=args.file_name, save_path=save_path)
	#2.  Slice it into dataframes and make a mapping file.
	output_path = args.output_dir
	if output_path[-1] != "/":
		output_path += "/"
	if merchant_or_subtype == "subtype":
		train_poor, test_poor, num_of_classes = subtype_preprocess(
			input_file, debit_or_credit, bank_or_card, output_path=output_path)
	else:
		raise Exception('!!!!')
	#3.  Use qlua to convert the files into training and testing sets.
	train_file = convert_csv_to_torch_7_binaries(train_poor)
	test_file = convert_csv_to_torch_7_binaries(test_poor)
	#4 Create a new configuration file based on the number of classes.
	create_new_configuration_file(num_of_classes, output_path, train_file, test_file)
	#5 Copy main.lua and data.lua to output directory.
	copy_file("meerkat/classification/lua/main.lua", output_path)
	copy_file("meerkat/classification/lua/data.lua", output_path)
	copy_file("meerkat/classification/lua/model.lua", output_path)
	copy_file("meerkat/classification/lua/train.lua", output_path)
	copy_file("meerkat/classification/lua/test.lua", output_path)
	#6 Excuete main.lua.
	execute_main_lua(output_path, "main.lua")

# The main program starts here if run from the command line.
if __name__ == "__main__":
	main_stream()

