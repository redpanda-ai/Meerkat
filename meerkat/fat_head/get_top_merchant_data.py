import sys
import os
import csv
import logging
import argparse
import shutil
import pandas as pd
from meerkat.classification.tools import (pull_from_s3, extract_tarball,
	check_new_input_file)

def select_merchant(input_file, chunksize, bank_or_card):
	dfs = []
	for chunk in pd.read_csv(input_file, chunksize=chunksize, error_bad_lines=False,
		encoding='utf-8', quoting=csv.QUOTE_NONE, na_filter=False, sep='|'):
		grouped = chunk.groupby('MERCHANT_NAME', as_index=False)
		groups = dict(list(grouped))
		if 'Starbucks' in groups.keys():
			dfs.append(groups['Starbucks'])
			logging.info(str(len(groups['Starbucks'])) + " transactions in current chunk")

	logging.info("start merging dataframes")
	merged = pd.concat(dfs, ignore_index=True)
	logging.info("finish merging dataframes")
	merged.to_csv('Starbucks_' + bank_or_card + '.csv', sep='|', index=False)

def parse_arguments(args):
	parser = argparse.ArgumentParser()
	parser.add_argument("bank_or_card")
	parser.add_argument("version_dir")
	parser.add_argument("--bucket", default="s3yodlee")
	args = parser.parse_args(args)
	return args

def main_process():
	logging.basicConfig(level=logging.INFO)
	args = parse_arguments(sys.argv[1:])
	bank_or_card = args.bank_or_card
	bucket = args.bucket
	version_dir = args.version_dir

	prefix = "meerkat/cnn/data/merchant/" + bank_or_card + "/" + version_dir + "/"
	extension = "tar.gz"
	tarball_name = "input.tar.gz"
	save_path = "meerkat/fat_head/data/input/"
	os.makedirs(save_path, exist_ok=True)
	pull_from_s3(bucket=bucket, prefix=prefix, extension=extension,
		file_name=tarball_name, save_path=save_path)
	extract_tarball(save_path + tarball_name, save_path)

	for file_name in os.listdir(save_path):
		if file_name.endswith(".csv"):
			logging.info("csv file at: " + save_path + file_name)
			selected_merchant_file = select_merchant(save_path + file_name, 1000000, bank_or_card)

	shutil.rmtree(save_path)

if __name__ == "__main__":
	main_process()

