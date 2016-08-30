import sys
import re
import os
import csv
import shutil
import pandas as pd
import logging
import yaml
import fileinput

from meerkat.classification.tools import (pull_from_s3, extract_tarball,
	check_new_input_file)

from .tools import get_etags, get_s3_file, remove_special_chars, get_grouped_dataframes
from .geomancer_module import GeomancerModule

logging.config.dictConfig(yaml.load(open('meerkat/geomancer/logging.yaml', 'r')))
logger = logging.getLogger('get_top_merchant_data')

def extract_clean_files(save_path, tarball_name, extension):
	"""This clears out a local directory and replaces files of a certain extension."""
	for root, dirs, files in os.walk(save_path):
		for f in files:
			if f.endswith(extension):
				os.unlink(os.path.join(root, f))
				logger.info("{0} is removed".format(f))
	extract_tarball(save_path + tarball_name, save_path)

class Worker(GeomancerModule):
	"""Contains methods and data pertaining to get top merchant data"""
	name = "top_merchant_data"
	def __init__(self, common_config, config):
		"""Constructor"""
		super(Worker, self).__init__(common_config, config)

	def main_process(self):
		bank_or_card = self.common_config["bank_or_card"]
		bucket = self.common_config["bucket"]
		#version_dir = self.config["version_dir"]

		prefix = "meerkat/geomancer/"
		#extension = "tar.gz"
		tarball_name = bank_or_card + "_transaction_sample.tar.gz"
		save_path = "meerkat/geomancer/data/input/" + bank_or_card + "/"
		os.makedirs(save_path, exist_ok=True)

		etags, etags_file = get_etags(save_path)
		logger.info("Synch-ing with S3")
		needs_to_be_downloaded = get_s3_file(bucket=bucket, prefix=prefix, file_name=tarball_name,
			save_path=save_path, etags=etags, etags_file=etags_file)
		logger.info("Synch-ed")

		if needs_to_be_downloaded:
			extract_clean_files(save_path, tarball_name, ".csv")

		tasks_prefix = "meerkat/geomancer/merchants/"
		for file_name in os.listdir(save_path):
			if file_name.endswith(".csv"):
				csv_file = save_path + file_name
				logger.info("csv file at: " + csv_file)

				pattern = re.compile("(\\|\\|)([^|]+?\=[^|]?)+?")
				for line in fileinput.input(csv_file, inplace=True):
					if pattern.search(line):
						new_line = pattern.sub("  \g<2>", line)
						print(new_line.rstrip())
					else:
						print(line.rstrip())
				csv_kwargs = { "chunksize": 1000, "error_bad_lines": False, "encoding": 'utf-8',
					"quoting": csv.QUOTE_NONE, "na_filter": False, "sep": "|", "activate_cnn": True,
					"cnn": bank_or_card + "_merchant" }

				existing_lines = int(sum(1 for line in open(csv_file)))
				target_lines = self.config["truncate_lines"]
				logger.info("Existing lines: {0}, Target lines: {1}".format(existing_lines, target_lines))
				if existing_lines >= target_lines:
					logger.info("Proceeding.")
				else:
					logger.info("Unzipping files.")
					extract_clean_files(save_path, tarball_name, ".csv")
					logger.info("Files unzipped.")

				#Add the ability to truncate lines from the input csv file
				if self.config["truncate_lines"] != -1:
					logger.info("Truncating to {0} lines.".format(self.config["truncate_lines"]))
					with open(csv_file, "r", encoding="utf-8") as input_file:
						with open(csv_file + ".temp", "w", encoding="utf-8") as output_file:
							for i in range(0, self.config["truncate_lines"]):
								output_file.write(input_file.readline())
					os.rename(csv_file + ".temp", csv_file)

				merchant_dataframes = get_grouped_dataframes(csv_file, 'MERCHANT_NAME',
					self.common_config["target_merchant_list"], **csv_kwargs)
				merchants = sorted(list(merchant_dataframes.keys()))
				self.common_config["target_merchant_list"] = merchants
				for merchant in merchants:
					formatted_merchant = remove_special_chars(merchant)
					os.makedirs(tasks_prefix + formatted_merchant, exist_ok=True)
					tasks = tasks_prefix + formatted_merchant + "/" + bank_or_card + "_tasks.csv"

					original_len = len(merchant_dataframes[merchant])
					merchant_dataframes[merchant].drop_duplicates(subset="plain_text_description", keep="first", inplace=True)
					merchant_dataframes[merchant].to_csv(tasks, sep='\t', index=False, quoting=csv.QUOTE_ALL)
					logger.info("Merchant {0}: {2} duplicate transactions; {1} unique transactions".format(merchant,
						len(merchant_dataframes[merchant]), original_len - len(merchant_dataframes[merchant])))
		return self.common_config

if __name__ == "__main__":
	logger.critical("You cannot run this from the command line, aborting.")
