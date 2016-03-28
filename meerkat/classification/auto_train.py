#!/usr/local/bin/python3.3

"""This script streamlines an entire CNN training process from pulling raw
data from s3 to applying the trained CNN to a test set and return
performance matrics.

@author: J. Andrew Key
@author: Jie Zhang
@author: Oscar Pan
@author: Matt Sevrens
"""

############################## USAGE ###############################################################
USAGE = """
usage: auto_train [-h] [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR]
                  [--credit_or_debit CREDIT_OR_DEBIT] [--bucket BUCKET] [-d]
                  [-v]
                  {subtype,merchant,category} {bank,card} train_file test_file
                  label_map

positional arguments:
  {subtype,merchant,category}
                        Pick a valid Classifier type
  {bank,card}           Is the model being trained on card or bank
                        transactions?
  train_file            Name of the training file to be pulled
  test_file             Name of the test file to be pulled
  label_map             Path of the label map to be pulled

optional arguments:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        Path of the directory containing input file
  --output_dir OUTPUT_DIR
                        Directory to write files to. Default is meerkat/data/
  --credit_or_debit CREDIT_OR_DEBIT
                        Is the model for credit or debit transactions?
  --bucket BUCKET       Input bucket name, default is s3yodlee
  -d, --debug           log at DEBUG level
  -v, --info            log at INFO leve
"""

EXAMPLE = """
time nohup python3 -m meerkat.classification.auto_train merchant bank merchant_bank_train.csv \
merchant_bank_test_03182016.csv bank_merchant_label_map.json -v &
"""

###################################################################################################

import argparse
import logging
import os

import tensorflow as tf

from meerkat.classification.tools import pull_from_s3
from meerkat.classification.tensorflow_cnn import build_graph, train_model, validate_config
from meerkat.classification.verify_data import load_json
from meerkat.tools.CNN_stats import main_process as apply_cnn

def parse_arguments():
	"""This function parses arguments from our command line."""

	parser = argparse.ArgumentParser("auto_train")

	# Help Text
	help_text = {
		"model_type" : "Pick a valid Classifier type",
		"bank_or_card": "Is the model being trained on card or bank transactions?",
		"train_file": "Name of the training file to be pulled",
		"test_file": "Name of the test file to be pulled",
		"label_map": "Path of the label map to be pulled",
		"output_dir": "Directory to write files to. Default is meerkat/data/",
		"credit_or_debit": "Is the model for credit or debit transactions?",
		"bucket": "Input bucket name, default is s3yodlee",
		"input_dir": "Path of the directory containing input file",
		"debug": "log at DEBUG level",
		"info": "log at INFO level"
	}

	choices = {
		"model_type": ["subtype", "merchant", "category"],
		"bank_or_card": ["bank", "card"]
	}

	# Required arugments
	parser.add_argument("model_type", help=help_text["model_type"], choices=choices["model_type"])
	parser.add_argument("bank_or_card", help=help_text["bank_or_card"], 
		choices=choices["bank_or_card"])
	parser.add_argument("train_file", help=help_text["train_file"])
	parser.add_argument("test_file", help=help_text["test_file"])
	parser.add_argument("label_map", default='', help=help_text["label_map"])

	# Optional arguments
	parser.add_argument("--input_dir", help=help_text["input_dir"], default='')
	parser.add_argument("--output_dir", help=help_text["output_dir"], default='')
	parser.add_argument("--credit_or_debit", default='', help=help_text["credit_or_debit"])
	parser.add_argument("--bucket", help=help_text["bucket"], default='s3yodlee')
	parser.add_argument("-d", "--debug", help=help_text["debug"], action="store_true")
	parser.add_argument("-v", "--info", help=help_text["info"], action="store_true")

	args = parser.parse_args()

	if args.model_type == 'subtype' and args.credit_or_debit == '':
		raise Exception('For subtype data you need to declare debit or credit.')
	if args.debug:
		logging.basicConfig(level=logging.DEBUG)
	elif args.info:
		logging.basicConfig(level=logging.INFO)

	return args

def auto_train():
	"""Run the automated training process"""

	args = parse_arguments()
	bucket = args.bucket
	bank_or_card = args.bank_or_card
	credit_or_debit = args.credit_or_debit
	model_type = args.model_type
	data_type = model_type + '_' + bank_or_card + '_' + credit_or_debit

	dir_paths = {
		'subtype_card_debit': 'data/subtype/card/debit/',
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

	s3_params = {"bucket": bucket, "prefix": prefix, "save_path": save_path}
	train_file = pull_from_s3(extension='.csv', file_name=args.train_file, **s3_params)
	test_file = pull_from_s3(extension='.csv', file_name=args.test_file, **s3_params)
	label_map = pull_from_s3(extension='.json', file_name=args.label_map, **s3_params)

	# Load and Modify Config
	config = validate_config("config/tf_cnn_config.json")
	config["dataset"] = train_file
	config["label_map"] = load_json(label_map)
	config["num_labels"] = len(config["label_map"].keys())
	config["ledger_entry"] = args.credit_or_debit
	config["model_type"] = args.model_type

	# Train the model
	graph, saver = build_graph(config)

	with tf.Session(graph=graph) as sess:
		tf.initialize_all_variables().run()
		best_model_path = train_model(config, graph, sess, saver)

	# Evaluate trained model using test set
	ground_truth_labels = {
		'merchant' : 'MERCHANT_NAME',
		'subtype' : 'PROPOSED_SUBTYPE'
	}

	args.model = best_model_path
	args.testdata = test_file
	args.label_map = label_map
	args.doc_key = 'DESCRIPTION_UNMASKED'
	args.secondary_doc_key = 'DESCRIPTION'
	args.label_key = ground_truth_labels[model_type]
	args.predicted_key = 'PREDICTED_CLASS'
	args.is_merchant = (model_type == 'merchant')
	args.fast_mode = True

	logging.warning('Apply the best CNN to test data and calculate performance metrics')
	apply_cnn(args)

	logging.warning('The whole streamline process has finished')

# The main program starts here if run from the command line.
if __name__ == "__main__":
	auto_train()
