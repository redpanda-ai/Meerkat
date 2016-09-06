#!/usr/lacal/bin/python3.3

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
import pandas as pd
import tensorflow as tf

from meerkat.various_tools import load_params, push_file_to_s3
from meerkat.longtail.rnn_classification_report import evaluate_model
from meerkat.longtail.bilstm_tagger import validate_config, preprocess, build_graph, run_session
from meerkat.classification.tools import check_new_input_file, pull_from_s3, extract_tarball, make_tarfile

#################################### USAGE ###############################################
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
	parser.add_argument("--config", help=help_text["config"], default="./meerkat/longtail/bilstm_config.json")
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

	exist_new_input, newest_version_dir, version = check_new_input_file(**s3_params)
	s3_params["prefix"] = newest_version_dir + "/"
	os.makedirs(save_path, exist_ok=True)

	if exist_new_input:
		logging.info("There exists new input data")
		input_file = pull_from_s3(extension=".tar.gz", file_name="input.tar.gz", **s3_params)
		extract_tarball(input_file, save_path)

		# Split data.csv into train.csv and test.csv
		df = pd.read_csv(save_path + "data.csv", sep="|")
		shuffled_df = df.reindex(np.random.permutation(df.index))
		percent = int(df.shape[0] * 0.9)
		shuffled_df[:percent].to_csv(save_path + "train.csv", header=True, index=False, sep="|")
		shuffled_df[percent:].to_csv(save_path + "test.csv", header=True, index=False, sep="|")

		train_file = save_path + "train.csv"
		test_file = save_path + "test.csv"
		push_file_to_s3(train_file, bucket, s3_params["prefix"])
		push_file_to_s3(test_file, bucket, s3_params["prefix"])
		logging.info("Push train.csv and test.csv to S3")

		config = load_params(args.config)
		config["dataset"] = train_file

		# Train RNN model
		config = validate_config(config)
		config = preprocess(config)
		graph, saver = build_graph(config)
		ckpt_model_file = run_session(config, graph, saver)
		final_model_path = "./meerkat/longtail/model/"

		# Tar model files and push model.tar.gz to s3
		model_tar_file = "./meerkat/longtail/model.tar.gz"
		make_tarfile(model_tar_file, final_model_path)
		push_file_to_s3(model_tar_file, bucket, s3_params["prefix"])
		logging.info("Push the model to S3")
		os.remove(model_tar_file)

		# Evaluate the model again test.csv
		args.data = test_file
		args.model = ckpt_model_file
		args.w2i = final_model_path + "w2i.json"
		evaluate_model(args)
		logging.info("Evalute the model on test data")

		# Push result files to s3
		result_files = glob.glob(save_path + "*.csv")
		for single_file in result_files:
			push_file_to_s3(single_file, bucket, s3_params["prefix"])
		logging.info("Push all the results to S3")

	logging.info("RNN training is done")

if __name__ == "__main__":
	"""The main training stream"""
	auto_train()
