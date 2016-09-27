"""This module will download the best RNN model from S3 to local instance

@author: Jie Zhang
"""

import os
import sys
import logging
import shutil
import argparse
import boto3
import pandas as pd
from meerkat.classification.tools import extract_tarball

#################################### USAGE ###############################################
# python3 -m meerkat.longtail.rnn_auto_load -v
# python3 -m meerkat.longtail.rnn_auto_load --bucket BUCKET_NAME -v
##########################################################################################

def parse_arguments(args):
	"""This function parses arguments from command line."""
	parser = argparse.ArgumentParser("rnn_auto_load")
	help_text = {
		"bucket": "Input bucket name, default is s3yodlee",
		"prefix": "Input s3 directory name, default is meerkat/rnn/data/",
		"save_path": "Save the best rnn model here, default is meerkat/classification/models/",
		"info": "log at INFO level"
	}

	# Optional arguments
	parser.add_argument("--bucket", help=help_text["bucket"], default="s3yodlee")
	parser.add_argument("--prefix", help=help_text["prefix"], default="meerkat/rnn/data/")
	parser.add_argument("--save_path", help=help_text["save_path"],
		default="./meerkat/classification/models/")
	parser.add_argument("-v", "--info", help=help_text["info"], action="store_true")

	args = parser.parse_args(args)
	if args.info:
		logging.basicConfig(level=logging.INFO)
	return args

def get_classification_accuracy(bucket, file_path):
	"""Read a classification report on S3 without downloading it"""
	client = boto3.client("s3")
	obj = client.get_object(Bucket=bucket, Key=file_path)
	df = pd.read_csv(obj["Body"])
	try:
		logging.info("The accuracy of {} is: {}".format(file_path, df["Accuracy"][0]))
		return df["Accuracy"][0]
	except KeyError:
		logging.info("The accuracy of {} is: 0.0".format(file_path))
		return 0.0

def get_valid_model_directories(bucket, prefix, target_file, model_name):
	"""Get all valid model directories which must have classification_report.csv and results.tar.gz"""
	valid_model_directories = []
	client = boto3.client("s3")

	for key in client.list_objects(Bucket=bucket, Prefix=prefix)["Contents"]:
		if key["Key"].endswith("/"):
			directory = key["Key"]
			try:
				_ = client.list_objects(Bucket=bucket, Prefix=directory + target_file)["Contents"]
				_ = client.list_objects(Bucket=bucket, Prefix=directory + model_name)["Contents"]
				valid_model_directories.append(directory)
			except KeyError:
				logging.info("{} has no valid {} and/or {}".format(directory, target_file, model_name))

	logging.info("Valid model directories are {}".format(valid_model_directories))
	return valid_model_directories

def download_file_from_s3(bucket, source_file_path, destination_file_path):
	"""Download a file from S3 to local instance"""
	client = boto3.resource("s3")
	client.meta.client.download_file(bucket, source_file_path, destination_file_path)
	logging.info("Downloaded {} to {}".format(bucket + "/" + source_file_path, destination_file_path))

def auto_load():
	"""Select the best RNN model and download it to local instance"""
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
	rnn_model_path = save_path + "rnn_model/"
	if os.path.exists(rnn_model_path):
		shutil.rmtree(rnn_model_path)
	os.mkdir(rnn_model_path)

	extract_tarball(save_path + model_name, save_path + "rnn_model/")
	logging.info("The best RNN model has been saved in {}".format(save_path + "rnn_model/"))
	logging.info("Removing " + save_path + model_name)
	os.remove(save_path + model_name)
	os.remove(rnn_model_path + "correct.csv")
	os.remove(rnn_model_path + "mislabeled.csv")

	logging.info("RNN auto_load is done")

if __name__ == "__main__":
	# Load the best RNN model automatically
	auto_load()
