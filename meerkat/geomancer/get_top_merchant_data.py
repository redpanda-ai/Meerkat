import sys
import os
import csv
import logging
import argparse
import shutil
import pandas as pd
from meerkat.classification.tools import (pull_from_s3, extract_tarball,
	check_new_input_file)

from .get_merchant_dictionaries import TARGET_MERCHANTS, get_merchant_dataframes
from .get_agg_data import get_s3_file, get_etags
from .tools import deduplicate_csv

def parse_arguments(args):
	parser = argparse.ArgumentParser()
	parser.add_argument("bank_or_card")
	parser.add_argument("version_dir")
	parser.add_argument("--bucket", default="s3yodlee")
	args = parser.parse_args(args)
	return args

def main_process():
	log_format = "%(asctime)s %(levelname)s: %(message)s"
	logging.basicConfig(format=log_format, level=logging.INFO)
	args = parse_arguments(sys.argv[1:])
	bank_or_card = args.bank_or_card
	bucket = args.bucket
	version_dir = args.version_dir

	prefix = "meerkat/cnn/data/merchant/" + bank_or_card + "/" + version_dir + "/"
	extension = "tar.gz"
	tarball_name = "input.tar.gz"
	save_path = "meerkat/geomancer/data/input/"
	os.makedirs(save_path, exist_ok=True)

	etags, etags_file = get_etags(save_path)
	logging.info("Synch-ing with S3")
	needs_to_be_downloaded = get_s3_file(bucket=bucket, prefix=prefix, file_name=tarball_name,
		save_path=save_path, etags=etags, etags_file=etags_file)
	logging.info("Synch-ed")

	if needs_to_be_downloaded:
		for root, dirs, files in os.walk(save_path):
			for f in files:
				if f.endswith(".csv"):
					os.unlink(os.path.join(root, f))
					logging.info("{0} is removed".format(f))
		extract_tarball(save_path + tarball_name, save_path)

	tasks_prefix = "meerkat/geomancer/merchants/"
	for file_name in os.listdir(save_path):
		if file_name.endswith(".csv"):
			csv_file = save_path + file_name
			logging.info("csv file at: " + csv_file)

			read_csv_kwargs = { "error_bad_lines": False, "encoding": 'utf-8',
				"quoting": csv.QUOTE_NONE, "na_filter": False, "sep": "|" }
			to_csv_kwargs = {"sep": "|"}
			dedup_csv_kwargs = {
				"read": read_csv_kwargs,
				"to": to_csv_kwargs
			}
			logging.info("Start deduplicate csv")
			deduplicate_csv(csv_file, "DESCRIPTION_UNMASKED", True, **dedup_csv_kwargs)
			logging.info("Finish deduplicate csv")

			csv_kwargs = { "chunksize": 1000, "error_bad_lines": False, "encoding": 'utf-8',
				"quoting": csv.QUOTE_NONE, "na_filter": False, "sep": "|", "activate_cnn": True}
			merchant_dataframes = get_merchant_dataframes(csv_file, 'MERCHANT_NAME', **csv_kwargs)
			merchants = sorted(list(merchant_dataframes.keys()))
			for merchant in merchants:
				os.makedirs(tasks_prefix + merchant, exist_ok=True)
				tasks = tasks_prefix + merchant + "/" + bank_or_card + "_tasks.csv"
				merchant_dataframes[merchant].to_csv(tasks, sep=',', index=False, quoting=csv.QUOTE_ALL)

if __name__ == "__main__":
	main_process()

