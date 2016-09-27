#!/usr/lacal/bin/python3.3
#pylint: disable=too-many-locals
#pylint: disable=too-many-statements

"""This script streamlines an entire SWS training process from pulling raw
data from s3 to applying the trained SWS to a test set and return
performance metrics.

@author: Jie Zhang
@author: Oscar Pan
"""
import os
import sys
import shutil
import logging
import argparse
import numpy as np

from meerkat.various_tools import load_params, push_file_to_s3, load_piped_dataframe
from meerkat.longtail.sws import validate_config, build_graph, run_session
from meerkat.classification.classification_report import main_process as evaluate_model
from meerkat.classification.tools import (check_new_input_file, pull_from_s3, extract_tarball,
		make_tarfile, check_file_exist_in_s3)

#################################### USAGE ###############################################
# nohup python3 -m meerkat.longtail.sws_auto_train &
# nohup python3 -m meerkat.longtail.sws_auto_train --bucket BUCKET_NAME &
##########################################################################################

def parse_arguments(args):
	"""This function parses arguments from command line."""
	parser = argparse.ArgumentParser("sws_auto_train")
	help_text = {
		"bucket": "Input bucket name, default is s3yodlee",
		"prefix": "Input s3 directory name, default is meerkat/sws/data/",
		"output_dir": "Input output directory, default is ./data/sws_stats/",
		"config": "Input config file name, default is ./meerkat/longtail/sws_config.json",
		"info": "Log at INFO level",
		"debug": "Log at DEBUG level"
	}

	# Optional arguments
	parser.add_argument("--bucket", help=help_text["bucket"], default="s3yodlee")
	parser.add_argument("--prefix", help=help_text["prefix"], default="meerkat/sws/data/")
	parser.add_argument("--output_dir", help=help_text["output_dir"], default="./data/sws_stats/")
	parser.add_argument("--config", help=help_text["config"],
			default="./meerkat/longtail/sws_config.json")
	parser.add_argument("-v", "--info", help=help_text["info"], action="store_true")
	parser.add_argument("-d", "--debug", help=help_text["debug"], action="store_true")

	args = parser.parse_args(args)
	if args.info:
		logging.basicConfig(level=logging.INFO)
	return args

def sws_auto_train():
	"""Run the automated training process"""
	args = parse_arguments(sys.argv[1:])
	bucket = args.bucket
	prefix = args.prefix + "/" * (args.prefix[-1] != "/")
	save_path = args.output_dir + "/" * (args.output_dir[-1] != "/")

	s3_params = {"bucket": bucket, "prefix": prefix, "save_path": save_path}

	exist_new_input, newest_version_dir, _ = check_new_input_file(**s3_params)
	s3_params["prefix"] = newest_version_dir + "/"

	if os.path.exists(save_path):
		shutil.rmtree(save_path)
	os.makedirs(save_path)

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
		logging.info(input_file + " removed.")

		# Split data.csv into train.csv and test.csv
		df = load_piped_dataframe(save_path + "data.csv", encoding="latin1")
		shuffled_df = df.reindex(np.random.permutation(df.index))
		percent = int(df.shape[0] * 0.9)
		shuffled_df[:percent].to_csv(save_path + "train.csv", header=True, index=False, sep="|")
		shuffled_df[percent:].to_csv(save_path + "test.csv", header=True, index=False, sep="|")

		train_file = save_path + "train.csv"
		test_file = save_path + "test.csv"

		preprocessed = save_path + "preprocessed.tar.gz"
		make_tarfile(preprocessed, save_path)
		push_file_to_s3(preprocessed, bucket, s3_params["prefix"])
		os.remove(preprocessed)
		os.remove(save_path + "data.csv")
		logging.info("preprocessed.tar.gz (data.csv, train.csv, test.csv) has been uploaded to S3")

		config = load_params(args.config)
		config["dataset"] = train_file

		# Train SWS model
		config = validate_config(config)
		graph, saver = build_graph(config)
		ckpt_model_file = run_session(config, graph, saver)
		results_path = "./meerkat/longtail/sws_model/"

		# Evaluate the model against test.csv
		logging.info("Evaluting the model on test data")
		args.data = test_file
		args.model = ckpt_model_file
		args.model_name = ""
		args.label = "Tagged_merchant_string"
		args.label_map = "./meerkat/longtail/sws_map.json"
		args.predict_key = "Should_search"
		args.fast_mode = False
		args.sws = True
		args.secdoc_key = "Description"
		evaluate_model(args)
		logging.info("The {} has been evaluated".format(test_file))

		# Update stats to S3
		stats = ["classification_report.csv", "confusion_matrix.csv", "correct.csv", "mislabeled.csv"]
		for single in stats:
			os.rename("./data/CNN_stats/" + single, save_path + single)

		for single in stats[0:2]:
			push_file_to_s3(save_path + single, bucket, s3_params["prefix"])
			logging.info("{} has been uploaded to S3".format(single))

		model_files = ["train.ckpt", "train.meta"]
		for single in model_files:
			os.rename(results_path + single, save_path + single)
			logging.info("{} has been moved from {} to {}".format(single, results_path, save_path))

		os.remove(train_file)
		os.remove(test_file)

		results = save_path + "results.tar.gz"
		make_tarfile(results, save_path)
		push_file_to_s3(results, bucket, s3_params["prefix"])
		logging.info("{} has been uploaded to S3".format(results))

		shutil.rmtree(results_path)
		shutil.rmtree(save_path)
		logging.info("{} has been deleted".format(results_path))
		logging.info("{} has been deleted".format(save_path))

	logging.info("SWS auto training is done")

if __name__ == "__main__":
	# The main training stream
	sws_auto_train()
