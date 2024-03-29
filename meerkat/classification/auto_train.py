#!/usr/local/bin/python3.3

"""This script streamlines an entire CNN training process from pulling raw
data from s3 to applying the trained CNN to a test set and return
performance matrics.

@author: J. Andrew Key
@author: Jie Zhang
@author: Oscar Pan
@author: Matt Sevrens
"""

import argparse
import logging
import os
import sys
import shutil

import tensorflow as tf

from meerkat.classification.split_data import main_split_data
from meerkat.classification.tools import (pull_from_s3, check_new_input_file,
	make_tarfile, copy_file, check_file_exist_in_s3, extract_tarball)
from meerkat.classification.tensorflow_cnn import build_graph, train_model, validate_config
from meerkat.classification.ensemble_cnns import build_graph as build_ensemble_graph
from meerkat.classification.ensemble_cnns import train_model as train_ensemble_model
from meerkat.classification.soft_target import main as get_soft_target
from meerkat.various_tools import load_params, safe_input, push_file_to_s3
from meerkat.classification.auto_load import get_single_file_from_tarball
from meerkat.classification.classification_report import main_process as apply_cnn

def parse_arguments(args):
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
		"ensemble": "Including this flag will use ensemble method.",
		"region": "Indicate a region"
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
	parser.add_argument("--region", help=help_text["region"], default='')
	parser.add_argument("-d", "--debug", help=help_text["debug"], action="store_true")
	parser.add_argument("-v", "--info", help=help_text["info"], action="store_true")

	args = parser.parse_args(args)

	if (args.model_type == 'subtype' or args.model_type == 'category') and args.credit_or_debit == '':
		raise Exception('You need to declare debit or credit.')
	if args.debug:
		logging.basicConfig(level=logging.DEBUG)
	elif args.info:
		logging.basicConfig(level=logging.INFO)

	return args

def auto_train():
	"""Run the automated training process"""

	args = parse_arguments(sys.argv[1:])
	bucket = args.bucket
	bank_or_card = args.bank_or_card
	credit_or_debit = args.credit_or_debit
	model_type = args.model_type
	if args.region: 
		args.region = 'region_' + args.region
	region = args.region
	data_type = model_type + '/' + bank_or_card
	if model_type != "merchant":
		data_type = data_type + '/' + credit_or_debit
	if region != '':
		data_type = data_type + '/' + region

	default_prefix = 'meerkat/cnn/data/'
	prefix = default_prefix + data_type + '/' if args.input_dir == '' else args.input_dir
	prefix = prefix + '/' * (prefix[-1] != '/')

	if args.output_dir == '':
		save_path = "./data/input/" + data_type
	else:
		save_path = args.output_dir + '/'*(args.output_dir[-1] != '/')

	s3_params = {"bucket": bucket, "prefix": prefix, "save_path": save_path}

	logging.info("s3 bucket: %s" % bucket)
	logging.info("s3 prefix: %s" % prefix)
	exist_new_input, newest_version_dir, version = check_new_input_file(**s3_params)
	s3_params["prefix"] = newest_version_dir + "/"

	if args.output_dir == '':
		save_path = save_path + '_' + version + '/'
		s3_params["save_path"] = save_path

	os.makedirs(save_path, exist_ok=True)

	exist_results_tarball = check_file_exist_in_s3("results.tar.gz", **s3_params)
	if exist_results_tarball:
		local_zip_file = pull_from_s3(extension='.tar.gz', file_name="results.tar.gz", **s3_params)
		try:
			_ = get_single_file_from_tarball(save_path, local_zip_file, ".ckpt", extract=False)

			valid_options = ["yes", "no"]
			while True:
				retrain_choice = safe_input(prompt="Model has already been trained. " +
					"Do you want to retrain the model? (yes/no): ")
				if retrain_choice in valid_options:
					break
				else:
					logging.critical("Not a valid option. Valid options are: yes or no.")

			if retrain_choice == "no":
				os.remove(local_zip_file)
				logging.info("Auto train ends")
				shutil.rmtree(save_path)
				return
			else:
				os.remove(local_zip_file)
				logging.info("Retrain the model")
		except:
			logging.critical("results.tar.gz is invalid. Retrain the model")

	if exist_new_input:
		logging.info("There exists new input data")
		args.bucket = s3_params["bucket"]
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
	config_dir = "meerkat/classification/config/"
	if args.ensemble:
		config = load_params(config_dir + "ensemble_cnns_config.json")
	else:
		config = load_params(config_dir + "default_tf_config.json")
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
		config["temperature"] = 8
		config["num_cnns"] = 1
		config["dataset"] = "./data/output/soft_target.csv"
		config["stopping_criterion"] = 3

		graph, saver = build_ensemble_graph(config)

		with tf.Session(graph=graph) as sess:
			tf.initialize_all_variables().run()
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
	args.fast_mode = False
	args.model_name = ''

	logging.warning('Apply the best CNN to test data and calculate performance metrics')
	apply_cnn(args=args)
	copy_file(best_model_path, tarball_directory)
	copy_file(best_model_path.replace(".ckpt", ".meta"), tarball_directory)
	make_tarfile("results.tar.gz", tarball_directory)
	logging.info("Uploading results.tar.gz to S3 {0}".format(s3_params["prefix"]))
	push_file_to_s3("results.tar.gz", bucket, s3_params["prefix"])
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

	if args.ensemble:
		ensemble_dir = "meerkat/classification/models/ensemble_cnns/"
		shutil.rmtree(ensemble_dir)
		logging.info("remove constituent CNNs at: {0}".format(ensemble_dir))
		os.remove(config["dataset"])
		logging.info("remove soft target training data at: "+config["dataset"])

	logging.warning('The whole streamline process has finished')

# The main program starts here if run from the command line.
if __name__ == "__main__":
	auto_train()
