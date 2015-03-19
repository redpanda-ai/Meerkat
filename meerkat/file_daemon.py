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
import paramiko
import re
import sys
import time

from boto.s3.connection import Location
from datetime import date, timedelta
from operator import itemgetter
from plumbum import SshMachine, BG

from .custom_exceptions import FileProblem, InvalidArguments


def launch_remote_producer(params, instance_ip, user, keyfile, item):
	logging.info("Launching: {0}".format(item))
	panel_name, file_name = item[0:2]
	log_name = "logs/" + panel_name + "." + file_name + ".log"
	if instance_ip not in params:
		params[instance_ip] = SshMachine(instance_ip, user=user, keyfile=keyfile)
	remote = params[instance_ip]
	command = remote["carriage"][panel_name][file_name][log_name]
	with remote.cwd("/root/git/Meerkat"):
		f = (command) & BG
	logging.info(("Launched"))

def get_parameters():
	"""Validates the command line arguments and loads a dict of params."""
	input_file, params = None, None
	if len(sys.argv) != 1:
		logging.debug("Supply the following arguments: config_file")
		raise InvalidArguments(msg="Incorrect number of arguments", expr=None)
	try:
		#TODO: Set to file.json
		input_file = open("config/daemon/file.json", encoding='utf-8')
		params = json.loads(input_file.read())
		input_file.close()
	except IOError:
		logging.error("Configuration file not found, aborting.")
		raise FileProblem(msg="Cannot find a valid configuration file.", expr=None)
	return params

def start():
	#This producer needs updating to deal with oddities:
	count, max_count = 1, 100000000
	sleep_time_sec = 60
	params = get_parameters()
	distribute_clients(params)
	while count < max_count:
		params["not_finished"] = []
		logging.info("Scan #{0} of #{1}".format(count, max_count))
		scan_locations(params)
		count += 1
		count_running_clients(params)
		dispatch_clients(params)
		logging.info("Resting for {0} seconds.".format(sleep_time_sec))
		time.sleep(sleep_time_sec)
	logging.info("Done.")

def distribute_clients(params):
	lp = params["launchpad"]
	instance_ips, username = lp["instance_ips"], lp["username"]
	key_filename, producer = lp["key_filename"], lp["producer"]
	producer_hash = lp["producer_hash"]
	command = "sha1sum " + producer
	hash = None
	ssh = paramiko.SSHClient()
	ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
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
	ssh.close()

def dispatch_clients(params):
	lp = params["launchpad"]
	instance_ips, username = lp["instance_ips"], lp["username"]
	key_filename = lp["key_filename"]
	running_pattern = re.compile("^(.*) (.*)$")
	ssh = paramiko.SSHClient()
	ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

	polling_command = "ps -ef|grep python3.3|grep -v grep|awk ' { print $11,$12 }'"
	#command_base = "cd /root/git/Meerkat/ && nohup python3.3 -m meerkat.file_producer "
	command_base = "(cd /root/git/Meerkat ; source /root/.bash_profile ; ./test.sh "
	total_slots = params["launchpad"]["per_instance_clients"]
	for instance_ip in instance_ips:
		logging.info("Counting running clients on {0}".format(instance_ip))
		ssh.connect(instance_ip, username=username, key_filename=key_filename)
		_, stdout, _ = ssh.exec_command(polling_command)
		process_count = 0
		in_progress = []
		for line in stdout.readlines():
			if running_pattern.search(line):
				process_count += 1
		remaining_slots = total_slots - process_count
		logging.info("{0} has {1} remaining slots".format(instance_ip, remaining_slots))
		for i in range(remaining_slots):
			if params["not_started"]:
				item = params["not_started"].pop()
				panel_name, panel_file = item[0:2]
				launch_remote_producer(params, instance_ip, username, key_filename, item)
	ssh.close()

def count_running_clients(params):
	"""Count the running clients."""
	polling_command = "ps -ef|grep python3.3|grep -v grep|awk ' { print $12,$13 }'"
	lp = params["launchpad"]
	instance_ips, username = lp["instance_ips"], lp["username"]
	key_filename = lp["key_filename"]
	running_pattern = re.compile("^(.*) (.*)$")
	params["in_progress"] = []
	ssh = paramiko.SSHClient()
	ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	for instance_ip in instance_ips:
		logging.info("Counting running clients on {0}".format(instance_ip))
		ssh.connect(instance_ip, username=username, key_filename=key_filename)
		_, stdout, _ = ssh.exec_command(polling_command)
		process_count = 0
		in_progress = []
		for line in stdout.readlines():
			if running_pattern.search(line):
				matches = running_pattern.search(line)
				process_count += 1
				panel_name = matches.group(1)
				panel_file = matches.group(2)
				in_progress.append((panel_name, panel_file, params[panel_name]))
				logging.info("Panel name: {0}, Panel file: {1}".format(panel_name, panel_file))
		logging.info("{0} processes found.".format(process_count))
		params["in_progress"].extend(in_progress)
	ssh.close()
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
	two_weeks_ago = str(date.today() - timedelta(days=14)).replace("-","")
	#Get the set of recent files
	recent_files = [ x for x in not_started if x[1][:8] >= two_weeks_ago ]
	#Get the set of older files
	older_files = [ x for x in not_started if x[1][:8] < two_weeks_ago ]
	#Sort the recent files by date(reversed)
	recent_files = sorted(recent_files, key=itemgetter(1), reverse=True)
	#Sort the older files, by priority
	older_files = sorted(older_files, key=itemgetter(2), reverse=True)
	sorted_not_started = []
	sorted_not_started.extend(older_files)
	sorted_not_started.extend(recent_files)

	params["not_started"] = sorted_not_started
	overall_report = {
		"files_not_yet_finished" : len(not_finished),
		"files_in_progress" : len(in_progress),
		"files_not_yet_started" : len(not_started),
		"files_daily_update": len(recent_files),
		"files_backlog": len(older_files)
	}
	report_timestamp = time.strftime("%c")
	local_report_file = params.get("local_report_file", "/dev/null")
	with open(local_report_file, 'w') as report_file:
		report_file.write(report_timestamp + "\n")
		for key in overall_report.keys():
			logging.info("{0}: {1}".format(key, overall_report[key]))
			#report_file.write(item + "\n")
		report_file.write(json.dumps(overall_report, sort_keys=True, indent=4,
				separators=(",", ": ")) + "\n")
		for report in params["report"]:
			report_file.write(json.dumps(report, sort_keys=True, indent=4,
				separators=(",", ": ")) + "\n")

def scan_locations(params):
	"""This function starts the new_daemon."""
	location_pairs = params["location_pairs"]
	params["s3_conn"] = boto.connect_s3()
	pair_priority = 0
	params["report"] = []
	for pair in location_pairs:
		name = pair["name"]
		params[name] = pair_priority
		logging.info("Compariing {0}".format(name))
		logging.info("Scanning\n\t{0}\n\t{1}".format(pair["src_location"], pair["dst_location"]))
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
	# Find files that are new in the destination S3 directory
	newer_src = [(name, k, pair_priority)
		for k in src_dict.keys()
		if k in dst_keys and src_dict[k][3] > dst_dict[k][3]]
	# Combine both lists
	total_list = not_in_dst
	total_list.extend(newer_src)
	# Ensure that params contains a 'not_finished' key
	if "not_finished" not in params:
		params["not_finished"] = []
	# Extend the list of "not_finished" files to include what we just discovered
	params["not_finished"].extend(total_list)
	# Log a quick report of what was found
#	if "report" not in params:
#		params["report"] = []
	my_report = { "name": name, "src" : len(src_dict), "dst" : len(dst_dict),
		"newer_src": len(newer_src), "not_in_dst": len(not_in_dst) }
	params["report"].append(my_report)
#	logging.info("Report: {0}".format(params["report"]))

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
	#MAIN PROGRAM
	logger = logging.getLogger('file_daemon')
	logger.setLevel(logging.DEBUG)
	handler = logging.StreamHandler()
	handler.setLevel(logging.DEBUG)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	logger.debug("File daemon")
	logging.basicConfig(format='%(asctime)s %(message)s', filename='/data/1/log/file_daemon.log', \
		level=logging.INFO)

	logging.info("Scanning module activated.")
	start()
