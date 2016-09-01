#!/usr/lacal/bin/python3.3

"""This script streamlines an entire RNN training process from pulling raw
data from s3 to applying the trained RNN to a test set and return
performance metrics.

@author: Oscar Pan
"""

import argparse
import logging
import os
import sys
import shutil

import tensorflow as tf

#################################### USAGE ###############################################
##########################################################################################

def parse_arguments(args):
	"""This function parses arguments from our command line."""

	parser = argparse.ArgumentParser("rnn_auto_train")

	help_text = {
		"--bucket": "Input bucket name, default is s3yodlee",
		"--prefix": "Input s3 directory name, default is meerkat/rnn/data/",
		"--output_dir": "Input output directory, default is ./data/RNN_stats/",
		"--config": "Input config file name, default is ./meerkat/longtail/bilstm_config.json"
	}

	# Required arguments
	# Optional arguments
	parser.add_argument("--bucket", help=help_text["bucket"], default="s3yodlee")
	parser.add_argument("--prefix", help=help_text["bucket"], default="meerkat/rnn/data/")
	parser.add_argument("--outupt_dir", help=help_text["output_dir"], default="./data/RNN_stats/")

	args = parser.parse_args(args)
	return args

def auto_train():
	"""Run the automated training process"""

	args = parse_arguments(sys.argv[1:])
	bucket = args.bucket
	prefix = args.prefix + "/" * (args.prefix[-1] != "/")
	save_path = args.output_dir + "/" * (args.output_dir[-1] != "/")

	s3_params = {"bucket": bucket, "prefix": prefix, "save_path": save_path}

	# TODO: mod check_new_input_file return
	exist_new_input, newest_version_dir, version = check_new_input_file(**s3_params)
	s3_params["prefix"] = newest_version_dir + "/"

	os.makedirs(save_path, exist_ok=True)

	exist_results_tarball = check_file_exist_in_s3("results.tar.gz", **s3_params)
	if exist_results_tarball:
		local_zip_file = pull_from_s3(extension=".tar.gz", file="results.tar.gz", **s3_params)
		try:
			_ = get_single_file_from_tarball(save_path, local_zip_file, "ckpt", extract=False)

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
		save_path = "./data/input/RNN/"
		config = load_params(args.config)

