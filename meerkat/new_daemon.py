import boto
import json
import logging
import re
import sys

from boto.s3.connection import Location
from .custom_exceptions import FileProblem, InvalidArguments

def get_parameters():
	"""Validates the command line arguments and loads a dict of params."""
	input_file, params = None, None
	if len(sys.argv) != 2:
		logging.debug("Supply the following arguments: config_file")
		raise InvalidArguments(msg="Incorrect number of arguments", expr=None)
	try:
		input_file = open(sys.argv[1], encoding='utf-8')
		params = json.loads(input_file.read())
		input_file.close()
	except IOError:
		logging.critical("%s not found, aborting.", sys.argv[1])
		raise FileProblem(msg="Cannot find a valid configuration file.", expr=None)
	return params

def start():
	"""This function starts the new_daemon."""
	params = get_parameters()
	location_pairs = params["location_pairs"]
	params["s3_conn"] = boto.connect_s3()

	for pair in location_pairs:
		logging.info("Compariing {0}".format(pair["name"]))
		logging.debug("Scanning\n\t{0}\n\t{1}".format(pair["src_location"], pair["dst_location"]))
		src_dict = scan_s3_location(params, pair["src_location"])
		#logging.info(src_dict)
		dst_dict = scan_s3_location(params, pair["dst_location"])
		update_pending_files(params, src_dict, dst_dict)

	logging.info("There are {0} pending files".format(len(params["pending_files"])))


def scan_s3_location(params, location):
	"""Scan a single s3 location, and build a dict of files and timestamps"""
	#TODO: Implement
	#logging.info("Location {0}".format(location))
	location_pattern = re.compile("^([^/]+)/(.*)$")
	matches = location_pattern.search(location)
	bucket_name = matches.group(1)
	directory = matches.group(2)
	logging.debug("Bucket: {0}, Directory {1}".format(bucket_name, directory))
	bucket = params["s3_conn"].get_bucket(bucket_name, Location.USWest2)
	result = {}
	filename_pattern = re.compile("^(.*)/(.+)$")
	for k in bucket.list(prefix=directory):
		file_name = filename_pattern.search(k.name).group(2)
		result[file_name] = (bucket_name, directory, k.name, k.last_modified)
	return result

def update_pending_files(params, src_dict, dst_dict):
	"""Update the dictionary of files that need to be processed."""
	dst_keys = dst_dict.keys()
	not_in_dst = [ k for k in src_dict.keys() if k not in dst_keys ]
	newer_src = [ k for k in src_dict.keys() if k in dst_keys and src_dict[k][3] > dst_dict[k][3] ]
	if "pending_files" not in params:
		params["pending_files"] = []
	params["pending_files"].extend(not_in_dst)
	params["pending_files"].extend(newer_src)
	logging.info("Src count {0}, dst count {1}".format(len(src_dict), len(dst_dict)))
	logging.info("Not in dst {0}, Newer src {1}".format(len(not_in_dst), len(newer_src)))

if __name__ == "__main__":
	#MAIN PROGRAM
	logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/new_daemon.log', \
		level=logging.INFO)
	logging.info("Toy module activated.")
	start()
