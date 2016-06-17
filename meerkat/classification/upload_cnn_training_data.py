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

# check files
def main_process():
	"""This is the whole process"""
	bucket = 's3yodlee'
	default_prefix = 'meerkat/cnn/data/'

	# upload the tar.gz file to s3
	dtime = get_utc_iso_timestamp()
	data_type = sys.argv[2].replace("_", "/")
	prefix = default_prefix + data_type + '/' + dtime + '/'

	csv_num, json_exit = 0, False
	for filename in os.listdir(sys.argv[1]):
		if filename.endswith('.csv'):
			csv_num += 1
		elif filename.endswith('.json'):
			if json_exit:
				print("should only have one json file")
				sys.exit()
			json_exit = True
		else:
			print("file %s is not csv or json file" %filename)
			sys.exit()
	if csv_num == 0:
		print("should at least one csv file")
		sys.exit()
	if not json_exit:
		print("should have one json file")
		sys.exit()
	print("files checking finished")

	# tar gz the files
	print("processing...")
	make_tarfile("input.tar.gz", sys.argv[1])
	print("files gziped")

	print("uploading to s3")
	push_file_to_s3('input.tar.gz', bucket, prefix)
	print("uploaded to s3")

	#remove the tar.gz file in local
	os.remove("input.tar.gz")

if __name__ == "__main__":
	main_process()
