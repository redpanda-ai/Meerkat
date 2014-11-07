import boto
import gzip
import paramiko
import re
import sys
import time

from datetime import datetime, date

from boto.s3.connection import Key, Location

def begin_processing_loop(some_container, date_pattern, s3_input_path):
	"""Fetches a list of input files to process from S3 and loops over them."""
	conn = boto.connect_s3()

	#Set destination details
	dst_bucket_name = "yodleeprivate"
	dst_s3_path_regex = re.compile("panels/meerkat/" + some_container +\
	"/(.*" + date_pattern + "_[^/]+)")
	dst_local_path = "/mnt/ephemeral/output/"
	dst_bucket = conn.get_bucket(dst_bucket_name, Location.USWest2)

	#Get the list of completed files (already proceseed)
	completed = {}
	for j in dst_bucket.list():
		if dst_s3_path_regex.search(j.key):
			completed[dst_s3_path_regex.search(j.key).group(1)] = j.size

	#Set source details
	src_bucket_name = "yodleeprivate"
	src_s3_path_regex = re.compile(s3_input_path + some_container +\
	"/(.*" + date_pattern + "_[^/]+)")
	src_local_path = "/mnt/ephemeral/input/"
	src_bucket = conn.get_bucket(src_bucket_name, Location.USWest2)

	#Get list of pending files (yet to be processed)
	pending_list, commands = [], []
	for k in src_bucket.list():
		if src_s3_path_regex.search(k.key):
			file_name = src_s3_path_regex.search(k.key).group(1)
			if file_name in completed:
				#Exclude files that have already been completed
				ratio = float(k.size) / completed[file_name]
				#Completed incorrectly
				if ratio >= 1.8:
					print("Completed Size, Source Size, Ratio: {0}, {1}, {2:.2f}".format(completed[file_name], k.size, ratio))
					print("Re-running {0}".format(file_name))
					pending_list.append((k, k.size))
			else:
				pending_list.append((k, k.size))
	#Reverse the pending list so that they are processed in reverse
	#chronological order
	last_date = None
	date_finder_regex = re.compile(".*(20[0-9]{6}[^_]*)_.*")
	for thing, item in reversed(pending_list):
		new_date = date_finder_regex.search(thing.key).group(1)
		if last_date != new_date:
			item = "cd /root/git/Meerkat && ./calendar_launcher.sh " + some_container + " " + new_date
			commands.append(item)
			last_date = new_date
	dst_s3_path = "panels/meerkat/" + some_container + "/"
	return commands

def poll_clients(my_stack, running_processes):
	clients = [ "172.31.16.210", "172.31.16.208", "172.31.16.209" ]
	rsa_private_key_file = "/root/.ssh/jkey.pem"
	big_command = "ps -ef|grep python3.3|grep -v grep|awk ' { print $12 }'"
	ssh = paramiko.SSHClient()
	ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	target_procs = 12
	for client in clients:
		ssh.connect(client, username="root", key_filename=rsa_private_key_file)
		print(client)
		stdin, stdout, stderr = ssh.exec_command(big_command)
		x = stdout.readlines()
		print("Processes are: ")
		process_count = 0
		for line in x:
			process_count += 1
			print(line.strip())
		print("There are {0} running processes".format(process_count))
		new_procs = target_procs - process_count
		for proc in range(0,new_procs):
			if my_stack:
				command = my_stack.pop()
				print(command)
				if command not in running_processes:
					print("Adding to running processes.")
					stdin, stdout, stderr = ssh.exec_command(command)
					running_processes[command] = datetime.now()
				else:
					print("Command already issued {0}".format(command))
			else:
				print("Stack is empty, finished for now.")
				ssh.close()
				return
		print("There are {0} running processes".format(process_count))
		ssh.close()
	ssh.close()
	print("Waiting 10 seconds.")
	time.sleep(10)

#MAIN PROGRAM
#Usage: python3.3 -m meerkat.dispatch <date_pattern>

date_pattern = sys.argv[1]
new_stack = []
running_processes = {}
print("Begin Program")
command_stack = begin_processing_loop("bank", date_pattern, "panels/meerkat_split/")
command_stack.extend(begin_processing_loop("card", date_pattern, "ctprocessed/gpanel/"))
start_time = datetime.now()
print("{0}".format(start_time))
goal_seconds = 3 * 24 * 60 * 60
new_time = datetime.now()
while True:
	print("Beginning loop")
	poll_clients(command_stack, running_processes)
	if not command_stack:
		print("Command stack empty, refilling")
		command_stack = begin_processing_loop("bank", date_pattern, "panels/meerkat_split/")
		command_stack.extend(begin_processing_loop("card", date_pattern, "ctprocessed/gpanel/"))
	new_time = datetime.now()
	duration = (new_time - start_time).total_seconds()
	print("{0} seconds have passed. We will stop at {1}".format(duration, goal_seconds))
	if duration > goal_seconds:
		print("Ending.")
		sys.exit()
