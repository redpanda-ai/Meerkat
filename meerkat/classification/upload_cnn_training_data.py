#!/usr/local/bin/python3.3

"""
Bundle a directory of files, that are limited to CSV and JSON files into
input.tar.gz and then upload it to an S3 bucket according to our rules.

created on April 4, 2016
@author: Feifei Zhu
"""
# python3 -m meerkat.classification.upload_cnn_training_data <source_dir> <s3_path_type>
# example: python3 -m meerkat.classification.upload_cnn_training_data tmp subtype_bank_debit

import os
import sys
from meerkat.classification.tools import get_utc_iso_timestamp, make_tarfile
from meerkat.various_tools import push_file_to_s3
import logging

def get_prefix():
	"""Get the prefix directory"""
	default_prefix = 'meerkat/cnn/data/'
	data_type = sys.argv[2]
	return default_prefix + data_type.replace("_", "/") + "/"

def check_file_existence():
	"""Check that there should be at lease one csv file and exactly one json file"""
	source_dir = sys.argv[1]
	csv_num, json_exit = 0, False
	for filename in os.listdir(source_dir):
		if filename.endswith('.csv'):
			csv_num += 1
		elif filename.endswith('.json'):
			if json_exit:
				logging.error("should only have one json file")
				sys.exit()
			json_exit = True
		else:
			logging.error("file %s is not csv or json file" %filename)
			sys.exit()
	if csv_num == 0:
		logging.error("should at least one csv file")
		sys.exit()
	if not json_exit:
		logging.error("should have one json file")
		sys.exit()
	logging.info("files checking finished")

def main_process():
	"""This is the whole process"""

	logging.basicConfig(level=logging.INFO)
	bucket = 's3yodlee'
	dtime = get_utc_iso_timestamp()
	prefix = get_prefix() + dtime + '/'

	check_file_existence()

	# tar gz the files
	logging.info("processing...")
	make_tarfile("input.tar.gz", sys.argv[1])
	logging.info("files gziped")

	# upload the tar.gz file to s3
	logging.info("uploading to s3")
	push_file_to_s3('input.tar.gz', bucket, prefix)
	logging.info("uploaded to s3")

	#remove the tar.gz file in local
	os.remove("input.tar.gz")

if __name__ == "__main__":
	main_process()
