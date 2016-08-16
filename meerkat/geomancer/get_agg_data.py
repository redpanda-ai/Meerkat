"""This module will sync local agg data with s3"""

import argparse
import boto3
import json
import logging
import os
import sys
import yaml

from meerkat.various_tools import load_params

logging.config.dictConfig(yaml.load(open('meerkat/geomancer/logging.yaml', 'r')))
logger = logging.getLogger('get_agg_data')

def get_etags(base_dir):
	"""Fetch local ETag values from a local file"""
	etags_file = base_dir + "etags.json"
	etags = {}
	if os.path.isfile(etags_file):
		logger.info("ETags found.")
		etags = load_params(etags_file)
	else:
		logger.info("Etags not found")
	return etags, etags_file

def get_s3_file(**kwargs):
	"""Load agg data from s3 to the local host"""

	client = boto3.client("s3")
	remote_file = kwargs["prefix"] + kwargs["file_name"]
	local_file = kwargs["save_path"] + kwargs["file_name"]

	local_file_exist = False
	if os.path.isfile(local_file):
		logger.info("local file {0} exists".format(local_file))
		local_file_exist = True
	else:
		logger.info("local file {0} not found".format(local_file))

	logger.debug(client.list_objects(Bucket=kwargs["bucket"],
		Prefix=remote_file))
	remote_etag = client.list_objects(Bucket=kwargs["bucket"],
		Prefix=remote_file)["Contents"][0]["ETag"]

	if local_file_exist:
		local_etag = None
		if remote_file in kwargs["etags"]:
			local_etag = kwargs["etags"][remote_file]

		logger.info("{0: <6} ETag is : {1}".format("Remote", remote_etag))
		logger.info("{0: <6} ETag is : {1}".format("Local", local_etag))

		#If the file is already local, skip downloading
		if local_etag == remote_etag:
			logger.info("Agg data exists locally no need to download")
			#File does not need to be downloaded
			return False

	logger.info("start downloading agg data from s3")
	client.download_file(kwargs["bucket"], remote_file, local_file)
	logger.info("Agg data file is downloaded at: " + local_file)

	etags = {}
	etags[remote_file] = remote_etag
	with open(kwargs["etags_file"], "w") as outfile:
		logger.info("Writing {0}".format(kwargs["etags_file"]))
		json.dump(etags, outfile)

	#File needs to be downloaded
	return True

class Worker:
	"""Contains methods and data pertaining to the creation and retrieval of AggData files"""
	def __init__(self, common_config, config):
		"""Constructor"""
		self.config = config
		self.config["bucket"] = common_config["bucket"]

	def main_process(self):
		"""Execute the main programe"""
		bucket = self.config["bucket"]
		prefix = self.config["prefix"]
		file_name = self.config["filename"]
		save_path = self.config["savepath"]
		os.makedirs(save_path, exist_ok=True)

		etags, etags_file = get_etags(save_path)

		needs_to_be_downloaded = get_s3_file(bucket=bucket, prefix=prefix, file_name=file_name,
			save_path=save_path, etags=etags, etags_file=etags_file)

if __name__ == "__main__":
	logger.critical("You cannot run this from the command line, aborting.")

