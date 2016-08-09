import sys
import re
import os
import csv
import argparse
import shutil
import pandas as pd
import logging
import yaml

from meerkat.classification.tools import (pull_from_s3, extract_tarball,
	check_new_input_file)

from .get_merchant_dictionaries import TARGET_MERCHANTS, get_merchant_dataframes
from .get_agg_data import get_s3_file, get_etags
from .tools import remove_special_chars

logging.config.dictConfig(yaml.load(open('meerkat/geomancer/logging.yaml', 'r')))
logger = logging.getLogger('get_top_merchant_data')

def parse_arguments(args):
	parser = argparse.ArgumentParser()
	parser.add_argument("bank_or_card")
	parser.add_argument("version_dir")
	parser.add_argument("--bucket", default="s3yodlee")
	parser.add_argument("--truncate_lines", default=-1, type=int)
	args = parser.parse_args(args)
	return args

def extract_clean_files(save_path, tarball_name, extension):
	"""This clears out a local directory and replaces files of a certain extension."""
	for root, dirs, files in os.walk(save_path):
		for f in files:
			if f.endswith(".csv"):
				os.unlink(os.path.join(root, f))
				logger.info("{0} is removed".format(f))
	extract_tarball(save_path + tarball_name, save_path)

def main_process():
	args = parse_arguments(sys.argv[1:])
	bank_or_card = args.bank_or_card
	bucket = args.bucket
	version_dir = args.version_dir

	prefix = "meerkat/cnn/data/merchant/" + bank_or_card + "/" + version_dir + "/"
	extension = "tar.gz"
	tarball_name = "input.tar.gz"
	save_path = "meerkat/geomancer/data/input/" + args.bank_or_card + "/"
	os.makedirs(save_path, exist_ok=True)

	etags, etags_file = get_etags(save_path)
	logger.info("Synch-ing with S3")
	needs_to_be_downloaded = get_s3_file(bucket=bucket, prefix=prefix, file_name=tarball_name,
		save_path=save_path, etags=etags, etags_file=etags_file)
	logger.info("Synch-ed")

	if needs_to_be_downloaded:
		extract_clean_files(save_path, tarball_name, "csv")

	tasks_prefix = "meerkat/geomancer/merchants/"
	for file_name in os.listdir(save_path):
		if file_name.endswith(".csv"):
			csv_file = save_path + file_name
			logger.info("csv file at: " + csv_file)
			csv_kwargs = { "chunksize": 1000, "error_bad_lines": False, "encoding": 'utf-8',
				"quoting": csv.QUOTE_NONE, "na_filter": False, "sep": "|", "activate_cnn": True}

			existing_lines = int(sum(1 for line in open(csv_file)))
			target_lines = args.truncate_lines
			logger.info("Existing lines: {0}, Target lines: {1}".format(existing_lines, target_lines))
			if existing_lines >= target_lines:
				logger.info("Proceeding.")
			else:
				logger.info("Unzipping files.")
				extract_clean_files(save_path, tarball_name, "csv")
				logger.info("Files unzipped.")

			#Add the ability to truncate lines from the input csv file
			if args.truncate_lines != -1:
				logger.info("Truncating to {0} lines.".format(args.truncate_lines))
				with open(csv_file, "r", encoding="utf-8") as input_file:
					with open(csv_file + ".temp", "w", encoding="utf-8") as output_file:
						for i in range(0, args.truncate_lines):
							output_file.write(input_file.readline())
				os.rename(csv_file + ".temp", csv_file)

			merchant_dataframes = get_merchant_dataframes(csv_file, 'MERCHANT_NAME', **csv_kwargs)
			merchants = sorted(list(merchant_dataframes.keys()))
			for merchant in merchants:
				formatted_merchant = remove_special_chars(merchant)
				os.makedirs(tasks_prefix + formatted_merchant, exist_ok=True)
				tasks = tasks_prefix + formatted_merchant + "/" + bank_or_card + "_tasks.csv"

				original_len = len(merchant_dataframes[merchant])
				merchant_dataframes[merchant].drop_duplicates(subset="DESCRIPTION_UNMASKED", keep="first", inplace=True)
				merchant_dataframes[merchant].to_csv(tasks, sep=',', index=False, quoting=csv.QUOTE_ALL)
				logger.info("Merchant {0}: {2} duplicate transactions; {1} unique transactions".format(merchant,
					len(merchant_dataframes[merchant]), original_len - len(merchant_dataframes[merchant])))

if __name__ == "__main__":
	main_process()

