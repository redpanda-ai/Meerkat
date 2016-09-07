#!/usr/lacal/bin/python3.3
#pylint: disable=too-many-locals
#pylint: disable=too-many-statements

"""This script streamlines an entire RNN training process from pulling raw
data from s3 to applying the trained RNN to a test set and return
performance metrics.

@author: Oscar Pan
@author: Jie Zhang
"""

import os
import sys
import glob
import shutil
import logging
import argparse

import numpy as np

from meerkat.various_tools import load_params, push_file_to_s3, load_piped_dataframe
from meerkat.longtail.rnn_classification_report import evaluate_model
from meerkat.longtail.bilstm_tagger import validate_config, preprocess, build_graph, run_session
from meerkat.classification.tools import (check_new_input_file, pull_from_s3, extract_tarball,
		make_tarfile, check_file_exist_in_s3)

#################################### USAGE ###############################################
# nohup python3 -m meerkat.longtail.rnn_auto_train &
# nohup python3 -m meerkat.longtail.rnn_auto_train --bucket BUCKET_NAME &
##########################################################################################

def parse_arguments(args):
	"""This function parses arguments from command line."""
	parser = argparse.ArgumentParser("rnn_auto_train")
	help_text = {
		"bucket": "Input bucket name, default is s3yodlee",
		"prefix": "Input s3 directory name, default is meerkat/rnn/data/",
		"output_dir": "Input output directory, default is ./data/RNN_stats/",
		"config": "Input config file name, default is ./meerkat/longtail/bilstm_config.json",
		"info": "log at INFO level"
	}

	# Optional arguments
	parser.add_argument("--bucket", help=help_text["bucket"], default="s3yodlee")
	parser.add_argument("--prefix", help=help_text["prefix"], default="meerkat/rnn/data/")
	parser.add_argument("--output_dir", help=help_text["output_dir"], default="./data/RNN_stats/")
	parser.add_argument("--config", help=help_text["config"],
			default="./meerkat/longtail/bilstm_config.json")
	parser.add_argument("-v", "--info", help=help_text["info"], action="store_true")

	args = parser.parse_args(args)
	if args.info:
		logging.basicConfig(level=logging.INFO)
	return args

def auto_train():
	"""Run the automated training process"""
	args = parse_arguments(sys.argv[1:])
	bucket = args.bucket
	prefix = args.prefix + "/" * (args.prefix[-1] != "/")
	save_path = args.output_dir + "/" * (args.output_dir[-1] != "/")

	s3_params = {"bucket": bucket, "prefix": prefix, "save_path": save_path}

	exist_new_input, newest_version_dir, _ = check_new_input_file(**s3_params)
	s3_params["prefix"] = newest_version_dir + "/"
	os.makedirs(save_path, exist_ok=True)

	# Model already exists in the newest directory
	if check_file_exist_in_s3("results.tar.gz", **s3_params):
		logging.info("Model already exists, please create a new directory and start a new training")
		sys.exit()

	# Start a new traing with input data
	if exist_new_input:
		logging.info("There exists new input data")
		input_file = pull_from_s3(extension=".tar.gz", file_name="input.tar.gz", **s3_params)
		extract_tarball(input_file, save_path)
		os.remove(input_file)
		logging.info(input_file+" removed.")

		# Split data.csv into train.csv and test.csv
		df = load_piped_dataframe(save_path+"data.csv", encoding="latin1")
		shuffled_df = df.reindex(np.random.permutation(df.index))
		percent = int(df.shape[0] * 0.9)
		shuffled_df[:percent].to_csv(save_path + "train.csv", header=True, index=False, sep="|")
		shuffled_df[percent:].to_csv(save_path + "test.csv", header=True, index=False, sep="|")

		train_file = save_path + "train.csv"
		test_file = save_path + "test.csv"

		config = load_params(args.config)
		config["dataset"] = train_file

		# Train RNN model
		config = validate_config(config)
		config = preprocess(config)
		graph, saver = build_graph(config)
		ckpt_model_file = run_session(config, graph, saver)
		results_path = "./meerkat/longtail/model/"


		# Evaluate the model against test.csv
		logging.info("Evalute the model on test data")
		args.data = test_file
		args.model = ckpt_model_file
		args.w2i = results_path + "w2i.json"
		evaluate_model(args)

		# Tar model and push results.tar.gz to s3
		tar_file = "./meerkat/longtail/results.tar.gz"
		os.rename("./data/RNN_stats/correct.csv", results_path+"correct.csv")
		os.rename("./data/RNN_stats/mislabeled.csv", results_path+"correct.csv")
		make_tarfile(tar_file, results_path)
		push_file_to_s3(tar_file, bucket, s3_params["prefix"])
		logging.info("Push the model to S3")
		os.remove(tar_file)
		logging.info(tar_file+" removed.")

		# Tar data, train and test.csv and push preprocessed.tar.gz to s3
		os.makedirs("./data/rnn_data_temp/")
		tar_file = "./data/preprocessed.tar.gz"
		os.rename("./data/RNN_stats/data.csv", "./data/rnn_data_temp/data.csv")
		os.rename("./data/RNN_stats/train.csv", "./data/rnn_data_temp/train.csv")
		os.rename("./data/RNN_stats/test.csv", "./data/rnn_data_temp/test.csv")
		make_tarfile(tar_file, "./data/rnn_data_temp/")
		push_file_to_s3(tar_file, bucket, s3_params["prefix"])
		logging.info("Push data to S3")
		os.remove(tar_file)
		logging.info(tar_file+" removed.")

		# Push the rest to s3
		result_files = glob.glob(save_path + "*.csv")
		for single_file in result_files:
			push_file_to_s3(single_file, bucket, s3_params["prefix"])
		logging.info("Push all the results to S3")
		shutil.rmtree(save_path)
		shutil.rmtree(results_path)
		shutil.rmtree("./data/rnn_data_temp/")

	logging.info("RNN auto training is done")

if __name__ == "__main__":
	# The main training stream
	auto_train()
