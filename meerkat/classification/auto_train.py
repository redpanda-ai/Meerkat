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
                  {subtype,merchant,category} {bank,card}

positional arguments:
  {subtype,merchant,category}
                        Pick a valid Classifier type
  {bank,card}           Is the model being trained on card or bank
                        transactions?

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
time nohup python3 -m meerkat.classification.auto_train merchant bank -v &
"""

###################################################################################################

import argparse
import logging
import os
import shutil

import tensorflow as tf

from meerkat.classification.split_data import main_split_data
from meerkat.classification.tools import (pull_from_s3, check_new_input_file,
	push_file_to_s3, make_tarfile, copy_file, check_file_exist_in_s3, extract_tarball)
from meerkat.classification.tensorflow_cnn import build_graph, train_model, validate_config
from meerkat.classification.ensemble_cnns import build_graph as build_ensemble_graph
from meerkat.classification.ensemble_cnns import train_model as train_ensemble_model
from meerkat.classification.soft_target import main as get_soft_target
from meerkat.various_tools import load_params, safe_input
from meerkat.classification.auto_load import get_single_file_from_tarball
from meerkat.classification.classification_report import main_process as apply_cnn

def parse_arguments():
	"""This function parses arguments from our command line."""

	parser = argparse.ArgumentParser("auto_train")

	# Help Text
	help_text = {
		"model_type" : "Pick a valid Classifier type",
		"bank_or_card": "Is the model being trained on card or bank transactions?",
		"output_dir": "Directory to write files to. Default is meerkat/data/",
		"credit_or_debit": "Is the model for credit or debit transactions?",
		"bucket": "Input bucket name, default is s3yodlee",
		"input_dir": "Path of the directory containing input file",
		"debug": "log at DEBUG level",
		"info": "log at INFO level",
		"ensemble": "Including this flag will use ensemble method."
	}

	choices = {
		"model_type": ["subtype", "merchant", "category"],
		"bank_or_card": ["bank", "card"]
	}

	# Required arugments
	parser.add_argument("model_type", help=help_text["model_type"], choices=choices["model_type"])
	parser.add_argument("bank_or_card", help=help_text["bank_or_card"], 
		choices=choices["bank_or_card"])

	# Optional arguments
	parser.add_argument("--ensemble", action="store_true", help=help_text["ensemble"])
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
	data_type = model_type + '_' + bank_or_card
	if model_type != "merchant":
		data_type = data_type + '_' + credit_or_debit

	dir_paths = {
		'subtype_card_debit': 'meerkat/cnn/data/subtype/card/debit/',
		'subtype_card_credit': 'meerkat/cnn/data/subtype/card/credit/',
		'subtype_bank_debit': 'meerkat/cnn/data/subtype/bank/debit/',
		'subtype_bank_credit': 'meerkat/cnn/data/subtype/bank/credit/',
		'merchant_bank': 'meerkat/cnn/data/merchant/bank/',
		'merchant_card': 'meerkat/cnn/data/merchant/card/',
		'category_bank_debit': 'meerkat/cnn/data/category/bank/debit/',
		'category_bank_credit': 'meerkat/cnn/data/category/bank/credit/',
		'category_card_debit': 'meerkat/cnn/data/category/card/debit/',
		'category_card_credit': 'meerkat/cnn/data/category/card/credit/'
	}

	prefix = dir_paths[data_type] if args.input_dir == '' else args.input_dir
	prefix = prefix + '/' * (prefix[-1] != '/')

	if args.output_dir == '':
		save_path = "./data/input/" + data_type
	else:
		save_path = args.output_dir + '/'*(args.output_dir[-1] != '/')

	s3_params = {"bucket": bucket, "prefix": prefix, "save_path": save_path}

	exist_new_input, newest_version_dir, version = check_new_input_file(**s3_params)
	s3_params["prefix"] = newest_version_dir + "/"

	if args.output_dir == '':
		save_path = save_path + '_' + version + '/'
		s3_params["save_path"] = save_path

	os.makedirs(save_path, exist_ok=True)

	exist_results_tarball = check_file_exist_in_s3("results.tar.gz", **s3_params)
	if exist_results_tarball:
		pull_from_s3(extension='.tar.gz', file_name="results.tar.gz", **s3_params)
		try:
			model_file = get_single_file_from_tarball(save_path + "results.tar.gz", ".ckpt")
			os.remove(model_file)

			valid_options = ["yes", "no"]
			while True:
				retrain_choice = safe_input(prompt="Model has already been trained. " +
					"Do you want to retrain the model? (yes/no): ")
				if retrain_choice in valid_options:
					break
				else:
					logging.critical("Not a valid option. Valid options are: yes or no.")

			if retrain_choice == "no":
				logging.info("Auto train ends")
				shutil.rmtree(save_path)
				return
			else:
				logging.info("Retrain the model")
		except:
			logging.critical("results.tar.gz is invalid. Retrain the model")
		os.remove(save_path + "results.tar.gz")

	if exist_new_input:
		logging.info("There exists new input data")
		args.input_dir = s3_params["prefix"]
		args.file_name = "input.tar.gz"
		args.train_size = 0.9
		main_split_data(args)
		save_path = save_path + 'preprocessed/'
	else:
		output_file = pull_from_s3(extension='.tar.gz', file_name="preprocessed.tar.gz", **s3_params)
		extract_tarball(output_file, save_path)

	train_file = save_path + "train.csv"
	test_file = save_path + "test.csv"
	label_map = save_path + "label_map.json"

	#copy the label_map.json file
	tarball_directory = "data/CNN_stats/"
	os.makedirs(tarball_directory, exist_ok=True)
	shutil.copyfile(label_map, tarball_directory + "label_map.json")

	# Load and Modify Config
	if args.ensemble:
		config = load_params("meerkat/classification/config/ensemble_cnns_config.json")
	else:
		config = load_params("meerkat/classification/config/default_tf_config.json")
	config["label_map"] = label_map
	config["dataset"] = train_file
	config["ledger_entry"] = args.credit_or_debit
	config["container"] = args.bank_or_card
	config["model_type"] = args.model_type
	config = validate_config(config)

	# Train the model
	if args.ensemble:
		graph, saver = build_ensemble_graph(config)

		with tf.Session(graph=graph) as sess:
			tf.initialize_all_variables().run()
			train_ensemble_model(config, graph, sess, saver)

		get_soft_target("meerkat/classification/models/ensemble_cnns/", train_file, label_map)
		config["soft_target"] = True
		config["num_cnns"] = 1
		config["datatest"] = "./data/output/soft_target.csv"

		graph, saver = build_ensemble_graph(config)

		with tf.Session(graph=graph) as sess:
			tf.initialize_all_variables().run
			best_model_path = train_ensemble_model(config, graph, sess, saver)

	else:
		graph, saver = build_graph(config)

		with tf.Session(graph=graph) as sess:
			tf.initialize_all_variables().run()
			best_model_path = train_model(config, graph, sess, saver)

	# Evaluate trained model using test set
	ground_truth_labels = {
		'category' : 'PROPOSED_CATEGORY',
		'merchant' : 'MERCHANT_NAME',
		'subtype' : 'PROPOSED_SUBTYPE'
	}

	args.model = best_model_path
	args.data = test_file
	args.label_map = label_map

	args.doc_key = 'DESCRIPTION_UNMASKED'
	args.secdoc_key = 'DESCRIPTION'
	args.label = ground_truth_labels[model_type]
	args.predict_key = 'PREDICTED_CLASS'
	args.is_merchant = (model_type == 'merchant')
	args.fast_mode = False
	if args.ensemble:
		args.model_name = "model1/cnn:0"

	logging.warning('Apply the best CNN to test data and calculate performance metrics')
	apply_cnn(args)
	copy_file("meerkat/classification/models/train.ckpt", tarball_directory)
	copy_file("meerkat/classification/models/train.meta", tarball_directory)
	make_tarfile("results.tar.gz", tarball_directory)
	logging.info("Uploading results.tar.gz to S3 {0}".format(s3_params["prefix"]))
	push_file_to_s3("results.tar.gz", "s3yodlee", s3_params["prefix"])
	logging.info("Upload results.tar.gz to S3 sucessfully.")

	#Clean up dirty files
	os.remove("results.tar.gz")
	logging.info("Local results.tar.gz removed.")
	for dirty_file in os.listdir(tarball_directory):
		file_path = os.path.join(tarball_directory, dirty_file)
		if os.path.isfile(file_path):
			os.unlink(file_path)
			logging.info("Local {0} removed.".format(file_path))

	if exist_new_input:
		remove_dir = save_path[0:save_path.rfind("preprocessed/")]
		shutil.rmtree(remove_dir)
	else:
		shutil.rmtree(save_path)
	logging.info("remove directory of preprocessed files at: {0}".format(save_path))

	logging.warning('The whole streamline process has finished')

# The main program starts here if run from the command line.
if __name__ == "__main__":
	auto_train()
