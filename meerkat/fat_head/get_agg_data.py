import sys
import os
import json
import logging
import argparse
import pandas as pd

from boto.s3 import connect_to_region
from boto.s3.key import Key
import boto3

from meerkat.various_tools import load_params

def parse_arguments(args):
	parser = argparse.ArgumentParser()
	parser.add_argument("--bucket", default="s3yodlee")
	parser.add_argument("--prefix", default="meerkat/fathead/aggdata/")
	args = parser.parse_args(args)
	return args

def get_etags():
	"""Fetches local ETag values from a local file."""
	etags = {}
	etags_file = "meerkat/fat_head/data/agg/etags.json"
	if os.path.isfile(etags_file):
		logging.info("ETags found.")
		etags = load_params(etags_file)
	else:
		logging.info("Etags not found, agg data will be downloaded.")
	return etags, etags_file

def load_agg_data(**kwargs):
	client = boto3.client("s3")
	remote_file = kwargs["prefix"] + kwargs["file_name"]
	local_file = kwargs["save_path"] + kwargs["file_name"]

	remote_etag = client.list_objects(Bucket=kwargs["bucket"],
		Prefix=remote_file)["Contents"][0]["ETag"]

	local_etag = None
	if remote_file in kwargs["etags"]:
		local_etag = kwargs["etags"][remote_file]

	logging.info("{0: <6} ETag is : {1}".format("Remote", remote_etag))
	logging.info("{0: <6} ETag is : {1}".format("Local", local_etag))

	#If the file is already local, skip downloading
	if local_etag == remote_etag:
		logging.info("Agg data exists locally no need to download")
		return

	logging.info("start downloading agg data from s3")
	client.download_file(kwargs["bucket"], remote_file, local_file)
	logging.info("Agg data file is downloaded at: " + local_file)

	etags = {}
	etags[remote_file] = remote_etag
	with open(kwargs["etags_file"], "w") as outfile:
		logging.info("Writing {0}".format(kwargs["etags_file"]))
		json.dump(etags, outfile)

def main_process():
	logging.basicConfig(level=logging.INFO)
	args = parse_arguments(sys.argv[1:])
	bucket = args.bucket
	prefix = args.prefix
	file_name = "All_Merchants.csv"
	save_path = "meerkat/fat_head/data/agg/"
	os.makedirs(save_path, exist_ok=True)

	etags, etags_file = get_etags()

	load_agg_data(bucket=bucket, prefix=prefix, file_name=file_name,
		save_path=save_path, etags=etags, etags_file=etags_file)

if __name__ == "__main__":
	main_process()

