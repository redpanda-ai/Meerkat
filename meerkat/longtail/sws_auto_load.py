"""This module will download the best SWS model from S3 to local instance

@author: Jie Zhang
@author: Oscar Pan
"""

import os
import sys
import logging
import shutil
import argparse
from meerkat.classification.tools import extract_tarball
from meerkat.longtail.rnn_auto_load import (get_classification_accuracy,
		get_valid_model_directories, download_file_from_s3)

#################################### USAGE ###############################################
# python3 -m meerkat.longtail.sws_auto_load -v
# python3 -m meerkat.longtail.sws_auto_load --bucket BUCKET_NAME -v
##########################################################################################

def parse_arguments(args):
	"""This function parses arguments from command line."""
	parser = argparse.ArgumentParser("sws_auto_load")
	help_text = {
		"bucket": "Input bucket name, default is s3yodlee",
		"prefix": "Input s3 directory name, default is meerkat/sws/data/",
		"save_path": "Save the best rnn model here, default is meerkat/classification/models/",
		"info": "log at INFO level"
	}

	# Optional arguments
	parser.add_argument("--bucket", help=help_text["bucket"], default="s3yodlee")
	parser.add_argument("--prefix", help=help_text["prefix"], default="meerkat/sws/data/")
	parser.add_argument("--save_path", help=help_text["save_path"],
		default="./meerkat/classification/models/")
	parser.add_argument("-v", "--info", help=help_text["info"], action="store_true")

	args = parser.parse_args(args)
	if args.info:
		logging.basicConfig(level=logging.INFO)
	return args

def sws_auto_load():
	"""Select the best SWS model and download it to local instance"""
	args = parse_arguments(sys.argv[1:])
	bucket = args.bucket
	prefix = args.prefix
	save_path = args.save_path + "/" * (args.save_path[-1] != "/")

	best_accuracy, best_model_prefix = 0.0, ""
	target_file, model_name = "classification_report.csv", "results.tar.gz"

	valid_model_directories = get_valid_model_directories(bucket, prefix, target_file, model_name)
	for key in valid_model_directories:
		report_file = key + target_file
		report_accuracy = get_classification_accuracy(args.bucket, report_file)
		if report_accuracy > best_accuracy:
			best_accuracy = report_accuracy
			best_model_prefix = key
	logging.info("The best model accuracy is {}".format(best_accuracy))
	logging.info("The best model path is {}".format(best_model_prefix))

	download_file_from_s3(args.bucket, best_model_prefix + model_name, save_path + model_name)
	sws_model_path = save_path + "sws_model/"
	if os.path.exists(sws_model_path):
		shutil.rmtree(sws_model_path)
	os.mkdir(sws_model_path)

	extract_tarball(save_path + model_name, sws_model_path)
	logging.info("The best sws model has been saved in {}".format(sws_model_path))
	logging.info("Removing " + save_path + model_name)
	os.remove(save_path + model_name)

	stats = ["classification_report.csv", "confusion_matrix.csv", "correct.csv", "mislabeled.csv"]
	for single in stats:
		os.remove(sws_model_path + single)

	logging.info("SWS auto_load is done")

if __name__ == "__main__":
	# Load the best SWS model automatically
	sws_auto_load()
