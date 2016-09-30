import os
import csv
import logging
import yaml

from meerkat.classification.tools import extract_tarball

from .tools import get_etags, get_s3_file, remove_special_chars, get_grouped_dataframes
from .geomancer_module import GeomancerModule

logging.config.dictConfig(yaml.load(open('meerkat/geomancer/logging.yaml', 'r')))
logger = logging.getLogger('get_top_merchant_data')

def extract_clean_files(save_path, tarball_name, extension):
	"""This clears out a local directory and replaces files of a certain extension."""
	for root, _, files in os.walk(save_path):
		for file_name in files:
			if file_name.endswith(extension):
				os.unlink(os.path.join(root, file_name))
				logger.info("{0} is removed".format(file_name))
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

		prefix = self.config["prefix"]
		tarball_name = bank_or_card + "_transaction_sample.tab.tar.gz"
		save_path = "meerkat/geomancer/data/input/" + bank_or_card + "/"
		os.makedirs(save_path, exist_ok=True)

		etags, etags_file = get_etags(save_path)
		logger.info("Synch-ing with S3")
		needs_to_be_downloaded = get_s3_file(bucket=bucket, prefix=prefix, file_name=tarball_name,
			save_path=save_path, etags=etags, etags_file=etags_file)
		logger.info("Synch-ed")

		if needs_to_be_downloaded:
			extract_clean_files(save_path, tarball_name, ".tab")

		tasks_prefix = "meerkat/geomancer/merchants/"
		for file_name in os.listdir(save_path):
			if file_name.endswith(".tab"):
				sample_file = save_path + file_name
				logger.info("Sample file at: " + sample_file)

				csv_kwargs = {"chunksize": 1000, "error_bad_lines": False, "encoding": 'utf-8',
					"quoting": csv.QUOTE_NONE, "na_filter": False, "sep": "\t", "activate_cnn": True,
					"cnn": bank_or_card + "_merchant"}

				sample = self.config["sample"]
				rebuild, limit = sample["rebuild"], sample.get("limit", None)
				#Destroy empty files
				current_file_size = int(sum(1 for line in open(sample_file)))
				if os.stat(sample_file).st_size == 0:
					logger.info("Destroying empty file.")
					rebuild = True
				# Rebuild if the pre-existing file is too small
				# adding 1 to the limit to account for header in file size
				elif limit and (limit + 1) > current_file_size:
					logger.info("Current file has only {0} records, rebuilding.".format(current_file_size - 1))
					rebuild = True
				# Rebuild if necessary
				if rebuild:
					logger.info("Unzipping files.")
					extract_clean_files(save_path, tarball_name, ".tab")
					logger.info("Files unzipped.")
				# Limit if necessary
				if limit:
					logger.info("Target to truncate to {0} lines".format(limit))
					lines_processed = 0
					found_header = False
					with open(sample_file, "r", encoding="utf-8") as input_file:
						with open(sample_file + ".temp", "w", encoding="utf-8") as output_file:
							for line in input_file:
								output_file.write(line)
								if not found_header:
									found_header = True
									continue
								lines_processed += 1
								if lines_processed % 10000 == 0:
									logger.info("Processed {0} lines".format(lines_processed * 10000))
								if lines_processed >= limit:
									logger.info("Truncated to limit: {0}".format(limit))
									break
							if lines_processed < limit:
								logger.warning("Sample file only contains {0} lines, however {1}"
									" lines were in the limit.".format(lines_processed, limit))
					os.rename(sample_file + ".temp", sample_file)

				#Everything is good

				merchant_dataframes = get_grouped_dataframes(sample_file, 'MERCHANT_NAME',
					self.common_config["target_merchant_list"], **csv_kwargs)
				merchants = sorted(list(merchant_dataframes.keys()))
				self.common_config["target_merchant_list"] = merchants
				for merchant in merchants:
					formatted_merchant = remove_special_chars(merchant)
					os.makedirs(tasks_prefix + formatted_merchant, exist_ok=True)
					tasks = tasks_prefix + formatted_merchant + "/" + bank_or_card + "_tasks.csv"

					original_len = len(merchant_dataframes[merchant])
					merchant_dataframes[merchant].drop_duplicates(subset="plain_text_description",
						keep="first", inplace=True)
					merchant_dataframes[merchant].to_csv(tasks, sep='\t', index=False, quoting=csv.QUOTE_ALL)
					logger.info("Merchant {0}: {2} duplicate transactions; {1} unique \
						transactions".format(merchant,
						len(merchant_dataframes[merchant]), original_len - len(merchant_dataframes[merchant])))
		return self.common_config

if __name__ == "__main__":
	logger.critical("You cannot run this from the command line, aborting.")
