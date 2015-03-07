import boto
import json
import logging
import paramiko
import re
import sys
import time

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
	count, max_count = 0, 1
	sleep_time_sec = 5
	params = get_parameters()
	ssh = paramiko.SSHClient()
	ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	params["launchpad"]["ssh"] = ssh
	distribute_clients(params)
	while count < max_count:
		params["pending_files"] = []
		logging.info("Scan #{0} of #{1}".format(count, max_count))
		scan_locations(params)
		count += 1
		count_running_clients(params)
		logging.info("Resting for {0} seconds.".format(sleep_time_sec))
		time.sleep(sleep_time_sec)

def distribute_clients(params):
	lp = params["launchpad"]
	ssh, instance_ips, username = lp["ssh"], lp["instance_ips"], lp["username"]
	key_filename, producer = lp["key_filename"], lp["producer"]
	producer_hash = lp["producer_hash"]
	command = "sha1sum " + producer
	hash = None
	for instance_ip in instance_ips:
		logging.info("Checking producer client on {0}".format(instance_ip))
		ssh.connect(instance_ip, username=username, key_filename=key_filename)
		_, stdout, _ = ssh.exec_command(command)
		for line in stdout.readlines():
			hash = line[:40]
		logging.info("SHA1 is {0}".format(hash))
		if hash != producer_hash:
			logging.critical("Sorry, producer is the wrong version.")
			sys.exit()

def count_running_clients(params):
	polling_command = "ps -ef|grep python3.3|grep -v grep|awk ' { print $12 }'"
	lp = params["launchpad"]
	ssh, instance_ips, username = lp["ssh"], lp["instance_ips"], lp["username"]
	key_filename = lp["key_filename"]
	for instance_ip in instance_ips:
		logging.info("Counting running clients on {0}".format(instance_ip))
		ssh.connect(instance_ip, username=username, key_filename=key_filename)
		_, stdout, _ = ssh.exec_command(polling_command)
		process_count = 0
		for line in stdout.readlines():
			process_count += 1
			logging.info(line.strip())
		logging.info("{0} processes found.".format(process_count))


def scan_locations(params):
	"""This function starts the new_daemon."""
	location_pairs = params["location_pairs"]
	params["s3_conn"] = boto.connect_s3()
	#Distribute the meerkat.producer if necessary
	#This producer needs updating to deal with oddities:
		#Strangely ordered columns
		#Strangely named columns

	for pair in location_pairs:
		logging.info("Compariing {0}".format(pair["name"]))
		logging.info("Scanning\n\t{0}\n\t{1}".format(pair["src_location"], pair["dst_location"]))
		src_dict = scan_s3_location(params, pair["src_location"])
		dst_dict = scan_s3_location(params, pair["dst_location"])
		update_pending_files(params, src_dict, dst_dict)

	logging.info("There are {0} pending files".format(len(params["pending_files"])))

def scan_s3_location(params, location):
	"""Scan a single s3 location, and build a dict of files and timestamps"""
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
	logging.debug("List of unprocessed files:")
	for item in not_in_dst:
		logging.debug(item)
	logging.debug("List of files that are newer at the source:")
	for item in newer_src:
		logging.debug(item)

if __name__ == "__main__":
	#MAIN PROGRAM
	logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/new_daemon.log', \
		level=logging.INFO)
	logging.info("Scanning module activated.")
	start()
