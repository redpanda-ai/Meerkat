import argparse
import logging
import os
import sys
import shutil

import tensorflow as tf

from meerkat.classification.split_data import main_split_data
from meerkat.classification.tools import (pull_from_s3, check_new_input_file,
	make_tarfile, copy_file, check_file_exist_in_s3, extract_tarball)
from meerkat.various_tools import load_params, safe_input, push_file_to_s3
from meerkat.classification.auto_load import get_single_file_from_tarball
from meerkat.classification.classification_report import main_process as apply_cnn
from meerkat.classification.tensorflow_cnn import validate_config

def download_data(model_type, bank_or_card, credit_or_debit):

	bucket = 's3yodlee'
	data_type = model_type + '/' + bank_or_card
	if model_type != "merchant":
		data_type = data_type + '/' + credit_or_debit

	default_prefix = 'meerkat/cnn/data/'
	prefix = default_prefix + data_type + '/'
	prefix = prefix + '/' * (prefix[-1] != '/')

	save_path = "./data/input/" + data_type

	s3_params = {"bucket": bucket, "prefix": prefix, "save_path": save_path}

	exist_new_input, newest_version_dir, version = check_new_input_file(**s3_params)
	s3_params["prefix"] = newest_version_dir + "/"

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
				logging.info("train ends")
				shutil.rmtree(save_path)
				sys.exit()
			else:
				os.remove(local_zip_file)
				logging.info("Retrain the model")
		except:
			logging.critical("results.tar.gz is invalid. Retrain the model")

	if exist_new_input:
		logging.info("There exists new input data")
		parser = argparse.ArgumentParser()
		parser.add_argument('--model_type', default=model_type)
		parser.add_argument('--bank_or_card', default=bank_or_card)
		parser.add_argument('--credit_or_debit', default=credit_or_debit)
		parser.add_argument('--bucket', default=s3_params["bucket"])
		parser.add_argument('--input_dir', default=s3_params["prefix"])
		parser.add_argument('--file_name', default="input.tar.gz")
		parser.add_argument('--train_size', default=0.9)
		args, _ = parser.parse_known_args()
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
	config = load_params(config_dir + "default_tf_config.json")
	config["label_map"] = label_map
	config["dataset"] = train_file
	config["ledger_entry"] = credit_or_debit
	config["container"] = bank_or_card
	config["model_type"] = model_type
	config = validate_config(config)

	return config, test_file, s3_params

def upload_result(config, best_model_path, test_file, s3_params):

# Evaluate trained model using test set
	ground_truth_labels = {
		'category' : 'PROPOSED_CATEGORY',
		'merchant' : 'MERCHANT_NAME',
		'subtype' : 'PROPOSED_SUBTYPE'
	}

	parser = argparse.ArgumentParser()
	parser.add_argument('--model', default=best_model_path)
	parser.add_argument('--data', default=test_file)
	parser.add_argument('--label_map', default=config["label_map"])
	parser.add_argument('--model_type', default=config["model_type"])
	parser.add_argument('--doc_key', default='DESCRIPTION_UNMASKED')
	parser.add_argument('--secdoc_key', default='DESCRIPTION')
	parser.add_argument('--label', default=ground_truth_labels[config['model_type']])
	parser.add_argument('--predict_key', default='PREDICTED_CLASS')
	parser.add_argument('--fast_mode', default=False)
	parser.add_argument('--model_name', default='')
	parser.add_argument('--debug', default=True)
	args, _ = parser.parse_known_args()
	logging.warning('Apply the best CNN to test data and calculate performance metrics')
	apply_cnn(args=args)

	bucket = 's3yodlee'
	tarball_directory = "data/CNN_stats/"

	copy_file(best_model_path, tarball_directory)
	copy_file(best_model_path.replace(".ckpt", ".meta"), tarball_directory)
	# copy_file("meerkat/classification/models/train.ckpt", tarball_directory)
	# copy_file("meerkat/classification/models/train.meta", tarball_directory)
	make_tarfile("results.tar.gz", tarball_directory)

	exist_results_tarball = check_file_exist_in_s3("results.tar.gz", **s3_params)
	if not exist_results_tarball:
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

	shutil.rmtree(s3_params["save_path"])
	logging.info("remove directory of preprocessed files at: {0}".format(s3_params["save_path"]))

	logging.warning('The whole streamline process has finished')


if __name__ == "__main__":
	download_data('subtype', 'bank', 'credit')
