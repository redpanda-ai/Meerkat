#!/usr/local/bin/python3.3

"""This module scans S3 directories looking for files to process.
One can configure its behavior within config/daemon/file.json

Created on Mar 18, 2015
@author: J. Andrew Key
"""
#################### USAGE ##########################

# python3.3 -m meerkat.file_daemon

#####################################################


import boto
import json
import logging
import re
import sys
import time

from boto.s3.connection import Location
from datetime import date, timedelta
from operator import itemgetter
from plumbum import local, SshMachine, BG
from plumbum.cmd import grep

from .custom_exceptions import FileProblem, InvalidArguments


def launch_remote_producer(params, instance_ip, item):
	"""Launches a file_producer client on one of the launchpad instances."""
	logging.info("Launching: {0}".format(item))
	panel_name, file_name = item[0:2]
	log_name = "logs/" + panel_name + "." + file_name + ".log"
	#Verify that we have a 'remote' connection tothe host
	verify_remote_instance(params, instance_ip)
	user = params["launchpad"]["username"]
	keyfile = params["launchpad"]["key_filename"]
	remote = params[instance_ip]
	command = remote["carriage"][panel_name][file_name][log_name]
	with remote.cwd("/root/git/Meerkat"):
		f = (command) & BG

def verify_remote_instance(params, instance_ip):
	"""Ensures that our params dict maintains an open remote connection."""
	if instance_ip not in params:
		user = params["launchpad"]["username"]
		keyfile = params["launchpad"]["key_filename"]
		params[instance_ip] = SshMachine(instance_ip, user=user, keyfile=keyfile)

def get_parameters():
	"""Validates the command line arguments and loads a dict of params."""
	input_file, params = None, None
	if len(sys.argv) != 1:
		logging.debug("Do not supply command line arguments.")
		raise InvalidArguments(msg="Incorrect number of arguments", expr=None)
	try:
		input_file = open("config/daemon/file.json", encoding='utf-8')
		params = json.loads(input_file.read())
		input_file.close()
	except IOError:
		logging.error("Configuration file not found, aborting.")
		raise FileProblem(msg="Cannot find a valid configuration file.", expr=None)
	return params

def begin_scanning_loop():
	"""Begins scanning each panel in S3 at regular intervals"""
	count, max_count = 1, 100000000
	sleep_time_sec = 60
	params = get_parameters()
	#distribute_clients(params)
	distribute_file(params, "/root/git/Meerkat/meerkat/file_producer.py")
	distribute_file(params, "/root/git/Meerkat/config/daemon/file.json")
	while count < max_count:
		params["not_finished"] = []
		logging.info("Scan #{0} of #{1}".format(count, max_count))
		scan_locations(params)
		count += 1
		get_files_in_progress(params)
		launch_remote_clients_into_available_slots(params)
		logging.info("Resting for {0} seconds.".format(sleep_time_sec))
		time.sleep(sleep_time_sec)
	logging.info("Done.")

def distribute_file(params, file_to_push):
	logging.info("Synchronizing {0}".format(file_to_push))
	launchpad = params["launchpad"]
	instance_ips, user = launchpad["instance_ips"], launchpad["username"]
	keyfile = launchpad["key_filename"]
	file_pattern = re.compile("^(.*/)([^/]+)$")
	matches = file_pattern.search(file_to_push)
	if matches == None:
		logging.error("file_to_push fails to match regex, aborting.")
		sys.exit()
	working_directory = matches.group(1)
	working_filename = matches.group(2)
	# Get the local hash
	local_hash = None
	command = local["sha1sum"][file_to_push]
	with local.cwd(working_directory):
		local_hash = command().split(" ")[0]
	#local_hash = launchpad["local_hash"]
	if not local_hash:
		logging.error("Unable to find a local file to hash, aborting")
		sys.exit()
	logging.info("File hash is: {0}".format(local_hash))
	# Check the remote file, distribute as necessary
	for instance_ip in instance_ips:
		logging.info("Checking file hash on {0}".format(instance_ip))
		# Get the remote hash
		remote_hash = None
		with SshMachine(instance_ip, user=user, keyfile=keyfile) as remote:
			command = remote["sha1sum"][file_to_push]
			with remote.cwd(working_directory):
				remote_hash = command().split(" ")[0]
				logging.info("Remote hash for {0} is {1}".format(instance_ip, remote_hash))
				# Upload the file, if the remote hash is incorrect
				if remote_hash != local_hash:
					remote.upload(file_to_push, working_directory)
					logging.info("Copied {0} to {1}.".format(file_to_push, instance_ip))

def get_files_in_progress(params):
	"""Scans for available 'slots' on the remote clients.  Should it find any, it
	then launches an instance of the file_producer into those empty slots"""
	lp = params["launchpad"]
	instance_ips = lp["instance_ips"]
	total_slots = lp["per_instance_clients"]
	params["in_progress"] = []
	#Loop through each launchpad host
	#write_local_report(params)
	logging.info("Counting running clients")
	for instance_ip in instance_ips:
		#logging.info("Counting running clients on {0}".format(instance_ip))
		process_count = 0
		#Ensure that we have a remote producer
		verify_remote_instance(params, instance_ip)
		#Count remote file producers
		remote = params[instance_ip]
		cmd = remote["ps"]["-ef"] | grep["python3.3"]
		cmd_result = cmd(retcode=None)
		cmd_split = cmd_result.split("\n")
		for line in cmd_split:
			splits = re.split('\s+', line)
			if len(splits) > 12:
				if splits[10] == "meerkat.file_producer":
					process_count += 1
					panel_name, panel_file = splits[11:13]
					params["in_progress"].append((panel_name, panel_file, params[panel_name]))
					#logging.info("Panel name: {0}, Panel file: {1}".format(panel_name, panel_file))
		#Calculate available slots
		remaining_slots = total_slots - process_count
		#logging.info("{0} has {1} remaining slots".format(instance_ip, remaining_slots))
		#Fill slots from list of files that are 'not_started'

		#Add the files in progress
		#for i in range(remaining_slots):
		#	if (("not_started" in params) and (params["not_started"])):
				#my_list = params["not_started"][:]
				#item = my_list.pop()
		#		item = params["not_started"].pop()
		#		panel_name, panel_file = item[0:2]
		#		params["in_progress"].append((panel_name, panel_file, params[panel_name]))

	#write_local_report(params)

def launch_remote_clients_into_available_slots(params):
	"""Scans for available 'slots' on the remote clients.  Should it find any, it
	then launches an instance of the file_producer into those empty slots"""
	lp = params["launchpad"]
	instance_ips = lp["instance_ips"]
	total_slots = lp["per_instance_clients"]
	#Loop through each launchpad host
	#write_local_report(params)
	for instance_ip in instance_ips:
		logging.info("Counting running clients on {0}".format(instance_ip))
		process_count = 0
		#Ensure that we have a remote producer
		verify_remote_instance(params, instance_ip)
		#Count remote file producers
		remote = params[instance_ip]
		cmd = remote["ps"]["-ef"] | grep["python3.3"]
		cmd_result = cmd(retcode=None)
		cmd_split = cmd_result.split("\n")
		for line in cmd_split:
			splits = re.split('\s+', line)
			if len(splits) > 12:
				if splits[10] == "meerkat.file_producer":
					process_count += 1
					panel_name, panel_file = splits[11:13]
					logging.info("Panel name: {0}, Panel file: {1}".format(panel_name, panel_file))
		#Calculate available slots
		remaining_slots = total_slots - process_count
		logging.info("{0} has {1} remaining slots".format(instance_ip, remaining_slots))
		#Fill slots from list of files that are 'not_started'
		for i in range(remaining_slots):
			if (("not_started" in params) and (params["not_started"])):
				item = params["not_started"].pop()
				panel_name, panel_file = item[0:2]
				launch_remote_producer(params, instance_ip, item)

	write_local_report(params)

def write_local_report(params):
	"""Write a report of the current status to the local file system"""
	#Valuable metrics
	not_finished = params["not_finished"]
	in_progress = params["in_progress"]

	not_started = list(set(not_finished) - set(in_progress))
	#This scheduler prefers daily update files to all others, regardless
	#of priority.
	#Get todays date minus 14 days
	three_weeks_ago = str(date.today() - timedelta(days=14)).replace("-","")
	#Get the set of recent files
	recent_files = [ x for x in not_started if x[1][:8] >= three_weeks_ago ]
	logging.warning("RECENT_FILES")
	logging.warning(recent_files)
	#Get the set of older files
	older_files = [ x for x in not_started if x[1][:8] < three_weeks_ago ]
	#Sort the recent files by date(reversed)
	recent_files = sorted(recent_files, key=itemgetter(1), reverse=True)
	#Sort the older files, by priority and then date(reversed)
	older_files = sorted(older_files, key=itemgetter(2,1), reverse=True)

	sorted_not_started = []
	sorted_not_started.extend(older_files)
	sorted_not_started.extend(recent_files)

	params["not_started"] = sorted_not_started
	overall_report = {
		"meta": 
		[
			{"timestamp": time.strftime("%c")},
			{"total_unfinished" : len(not_finished)},
			{"_in_progress" : len(in_progress)},
			{"_backlog" : len(not_started)},
			{"__daily": len(recent_files)},
			{"__older": len(older_files)}
		],
		"panels": []
	}
	for report in params["report"]:
		overall_report["panels"].append(report)
	report_timestamp = time.strftime("%c")
	local_report_file = params.get("local_report_file", "/dev/null")
	full_report = json.dumps(overall_report, sort_keys=True, indent=4,
		separators=(",", ": "))
	logging.info(full_report)
	with open(local_report_file, 'w') as report_file:
		report_file.write(full_report)

def scan_locations(params):
	"""This function starts the new_daemon."""
	location_pairs = params["location_pairs"]
	params["s3_conn"] = boto.connect_s3()
	pair_priority = 0
	params["report"] = []
	for pair in location_pairs:
		name = pair["name"]
		params[name] = pair_priority
		logging.info("Comparing {0}".format(name))
		#logging.info("Scanning\n\t{0}\n\t{1}".format(pair["src_location"], pair["dst_location"]))
		src_dict = scan_s3_location(params, pair["src_location"])
		dst_dict = scan_s3_location(params, pair["dst_location"])
		update_pending_files(params, name, src_dict, dst_dict, pair_priority)
		pair_priority += 1

	logging.info("There are {0} pending files".format(len(params["not_finished"])))

def scan_s3_location(params, location):
	"""Scan a single s3 location, and build a dict of files and timestamps"""
	location_pattern = re.compile("^([^/]+)/(.*)$")
	matches = location_pattern.search(location)
	bucket_name = matches.group(1)
	directory = matches.group(2)
	logging.debug("Bucket: {0}, Directory {1}".format(bucket_name, directory))
	bucket = params["s3_conn"].get_bucket(bucket_name, Location.USWest2)
	result = {}
	filename_pattern = re.compile("^(.*)/(.+.txt.gz)$")
	for k in bucket.list(prefix=directory):
		if filename_pattern.search(k.name):
			file_name = filename_pattern.search(k.name).group(2)
			result[file_name] = (bucket_name, directory, k.name, k.last_modified)
	return result

def update_pending_files(params, name, src_dict, dst_dict, pair_priority):
	"""Update the dictionary of files that need to be processed."""
	# Look at the files at the S3 destination directory
	dst_keys = dst_dict.keys()
	# Find files not in destination S3 directory
	not_in_dst = [(name, k, pair_priority)
		for k in src_dict.keys()
		if k not in dst_keys]
	# Reprocess newer files
	if params["reprocess"]:
		newer_src = [(name, k, pair_priority)
			for k in src_dict.keys()
			if k in dst_keys and src_dict[k][3] > dst_dict[k][3]]
	else:
		newer_src = []
	# Combine both lists
	total_list = not_in_dst
	total_list.extend(newer_src)
	# Ensure that params contains a 'not_finished' key
	if "not_finished" not in params:
		params["not_finished"] = []
	# Extend the list of "not_finished" files to include what we just discovered
	params["not_finished"].extend(total_list)
	# Log a quick report of what was found
	my_report = { "name": name, "src" : len(src_dict), "dst" : len(dst_dict),
		"newer_src": len(newer_src), "not_in_dst": len(not_in_dst) }
	params["report"].append(my_report)
	logging.info("Src count {0}, dst count {1}".format(len(src_dict), len(dst_dict)))
	logging.info("Not in dst {0}, Newer src {1}".format(len(not_in_dst), len(newer_src)))
	logging.debug("List of unprocessed files:")
	# Log a more involved report if the log level is debug
	for item in not_in_dst:
		logging.debug(item)
	logging.debug("List of files that are newer at the source:")
	for item in newer_src:
		logging.debug(item)

if __name__ == "__main__":
	# MAIN PROGRAM
	logger = logging.getLogger('file_daemon')
	logger.setLevel(logging.DEBUG)
	handler = logging.StreamHandler()
	handler.setLevel(logging.DEBUG)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	logger.debug("File daemon")
	logging.basicConfig(format='%(asctime)s %(message)s', filename='/var/log/file_daemon_info.log', \
		level=logging.INFO)

	logging.info("Scanning module activated.")
	begin_scanning_loop()
