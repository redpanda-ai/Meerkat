import boto
import json
import logging
import paramiko
import re
import sys
import time

from boto.s3.connection import Location
from .custom_exceptions import FileProblem, InvalidArguments
from datetime import date, timedelta
from plumbum import SshMachine, BG

from operator import itemgetter

#Usage python3.3 -m meerkat.file_daemon

def launch_remote_producer(params, instance_ip, user, keyfile, item):
	logging.info("Launching: {0}".format(item))
	panel_name, file_name = item[0:2]
	log_name = "logs/" + panel_name + "." + file_name + ".log"
	if instance_ip not in params:
		params[instance_ip] = SshMachine(instance_ip, user=user, keyfile=keyfile)
	remote = params[instance_ip]
	command = remote["carriage"][panel_name][file_name][log_name]
	with remote.cwd("/root/git/Meerkat"):
		pass
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
				#logging.info("Launching {0}, which is {1}".format(i, item))
				panel_name, panel_file = item[0:2]
				#command = command_base + item[0] + " " + item[1] + " " + item[0] + "." + item[1] + ".log)"
				#logging.info("Command: {0}".format(command))
				#print(command)
				launch_remote_producer(params, instance_ip, username, key_filename, item)
				#_, _, _ = ssh.exec_command(command)
				#logging.info(stdout)
				#logging.info("Command: {0}".format(command))
	ssh.close()

def count_running_clients(params):
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
	#Valuable metrics
	not_finished = params["not_finished"]
	in_progress = params["in_progress"]
	not_started = list(set(not_finished) - set(in_progress))

	#Now for new scheduling
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
	logging.info("Number of files not yet finished: {0}".format(len(not_finished)))
	logging.info("Number of files in progress: {0}".format(len(in_progress)))
	logging.info("Number of files not yet started: {0}".format(len(not_started)))
	logging.info("Number of recent files not yet started {0}".format(len(recent_files)))
	logging.info("Number of older files not yet started {0}".format(len(older_files)))
	logging.info("Recent files:\n{0}".format(recent_files))
	#logging.info("Not started:\n {0}".format(not_started))

def scan_locations(params):
	"""This function starts the new_daemon."""
	location_pairs = params["location_pairs"]
	params["s3_conn"] = boto.connect_s3()
	pair_priority = 0
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
	dst_keys = dst_dict.keys()
	not_in_dst = [ (name, k, pair_priority) for k in src_dict.keys() if k not in dst_keys ]
	newer_src = [ (name, k, pair_priority) for k in src_dict.keys() if k in dst_keys and src_dict[k][3] > dst_dict[k][3] ]
	if "not_finished" not in params:
		params["not_finished"] = []
	#TODO: Verify that reverse works
	total_list = []
	total_list.extend(not_in_dst)
	total_list.extend(newer_src)

	params["not_finished"].extend(total_list)
	#params["not_finished"].extend(my_not_in_dst)
	#params["not_finished"].extend(newer_src)
	logging.info("Src count {0}, dst count {1}".format(len(src_dict), len(dst_dict)))
	logging.info("Not in dst {0}, Newer src {1}".format(len(not_in_dst), len(newer_src)))
	logging.debug("List of unprocessed files:")
	for item in not_in_dst:
		logging.info(item)
	logging.debug("List of files that are newer at the source:")
	for item in newer_src:
		logging.info(item)

if __name__ == "__main__":
	#MAIN PROGRAM
	logger = logging.getLogger('file_daemon')
	logger.setLevel(logging.DEBUG)
	ch = logging.StreamHandler()
	ch.setLevel(logging.DEBUG)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	ch.setFormatter(formatter)
	logger.addHandler(ch)
	logger.debug("File daemon")
	logging.basicConfig(format='%(asctime)s %(message)s', filename='/data/1/log/file_daemon.log', \
		level=logging.INFO)

	logging.info("Scanning module activated.")
	start()
