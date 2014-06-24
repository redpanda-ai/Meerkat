#!/usr/local/bin/python3.3

"""This module loads and processes transactions
in preparation for analysis and search though
our structured ElasticSearch merchant index. 

Created on Dec 9, 2013
@author: J. Andrew Key
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3.3 -m meerkat [config_file]
# python3.3 -m meerkat config/users.json

#####################################################

import csv
import datetime
import collections
import gzip
import json
import logging
import os
import re
import queue
import sys

import boto
from boto.s3.connection import Location, S3Connection

from .custom_exceptions import InvalidArguments, Misconfiguration
from .description_consumer import DescriptionConsumer
from .binary_classifier.load import select_model
from .various_tools import load_dict_list, safely_remove_file, split_csv, merge_split_files
from .accuracy import test_accuracy, print_results, speed_tests

def get_desc_queue(filename, params, classifier):
	"""Opens a file of descriptions, one per line, and load a description
	queue."""

	encoding = None
	physical, non_physical, atm = [], [], []
	users = collections.defaultdict(list)
	desc_queue = queue.Queue()

	try:
		encoding = params["input"]["encoding"]
		delimiter = params["input"].get("delimiter", "|")
		transactions = load_dict_list(filename, encoding=encoding,\
			delimiter=delimiter)
	except IOError:
		logging.critical("Input file: %s"
			" cannot be found. Terminating", filename)
		sys.exit()

	# Run Binary Classifier
	for transaction in transactions:
		transaction['factual_id'] = ""
		description = transaction["DESCRIPTION_UNMASKED"]
		prediction = classifier(description)
		transaction["IS_PHYSICAL_TRANSACTION"] = prediction
		print(str(description.encode("utf-8")) + ": " + str(prediction.encode("utf-8")))
		if prediction == "1":
			physical.append(transaction)
		elif prediction == "0":
			non_physical.append(transaction)
			logging.info("NON-PHYSICAL: %s", description)
		elif prediction == "2":
			physical.append(transaction)
			atm.append(transaction)

	# Split into user buckets
	for row in physical:
		user = row.get("UNIQUE_MEM_ID", "")
		users[user].append(row)

	# Add Users to Queue
	for key, _ in users.items():
		desc_queue.put(users[key])

	atm_percent = (len(atm) / len(transactions)) * 100
	non_physical_percent = (len(non_physical) / len(transactions)) * 100
	physical_percent = (len(physical) / len(transactions)) * 100

	print("")
	print("PHYSICAL: ", round(physical_percent, 2), "%")
	print("NON-PHYSICAL: ", round(non_physical_percent, 2), "%")
	print("ATM: ", round(atm_percent, 2), "%")
	print("")

	return desc_queue, non_physical

def initialize():
	"""Validates the command line arguments."""
	input_file, params = None, None

	if len(sys.argv) != 2:
		usage()
		raise InvalidArguments(msg="Incorrect number of arguments", expr=None)

	try:
		input_file = open(sys.argv[1], encoding='utf-8')
		params = json.loads(input_file.read())
		input_file.close()
	except IOError:
		logging.error("%s not found, aborting.", sys.argv[1])
		sys.exit()

	return params

def load_hyperparameters(params):
	"""Attempts to load parameter key"""
	hyperparameters = None
	try:
		input_file = open(params["input"]["hyperparameters"], encoding='utf-8')
		hyperparameters = json.loads(input_file.read())
		input_file.close()
	except IOError:
		logging.error("%s not found, aborting.", params["input"]["hyperparameters"])
		sys.exit()
	return hyperparameters

def queue_to_list(result_queue):
	"""Converts queue to list"""
	result_list = []
	while result_queue.qsize() > 0:
		try:
			result_list.append(result_queue.get())
			result_queue.task_done()

		except queue.Empty:
			break
	result_queue.join()
	return result_list

def run_meerkat(params, desc_queue, hyperparameters, non_physical, split):
	"""Opens a number of threads to process the descriptions queue."""

	# Run the Classifier
	consumer_threads = params.get("concurrency", 8)
	result_queue = queue.Queue()

	for i in range(consumer_threads):
		new_consumer = DescriptionConsumer(i, params, desc_queue,\
			result_queue, hyperparameters)
		new_consumer.setDaemon(True)
		new_consumer.start()

	desc_queue.join()

	# Convert queue to list
	result_list = queue_to_list(result_queue)

	# Writing to an output file, if necessary.
	if "file" in params["output"] and "format" in params["output"]["file"]\
	and params["output"]["file"]["format"] in ["csv", "json"]:
		write_output_to_file(params, result_list, non_physical, split)
	else:
		logging.critical("Not configured for file output.")

	# Test Accuracy
	#accuracy_results = test_accuracy(params, result_list=result_list, non_physical_trans=non_physical)
	#print_results(accuracy_results)

	# Shutdown Loggers
	logging.shutdown()

	#return accuracy_results

def usage():
	"""Shows the user which parameters to send into the program."""
	result = "Usage:\n\t<path_to_json_format_config_file>"
	logging.error(result)
	return result

def validate_params(params):
	"""Ensures that the correct parameters are supplied."""

	mandatory_keys = ["elasticsearch", "concurrency", "input", "logging"]

	# Ensure Mandatory Keys are in Config
	for key in mandatory_keys:
		if key not in params:
			raise Misconfiguration(msg="Misconfiguration: missing key, '" + key + "'", expr=None)

	if params["concurrency"] <= 0:
		raise Misconfiguration(msg="Misconfiguration: 'concurrency' must be a positive integer", expr=None)

	if "hyperparameters" not in params["input"]:
		params["input"]["hyperparameters"] = "config/hyperparameters/default.json"

	# Ensure Other Various Parameters Available
	if "index" not in params["elasticsearch"]:
		raise Misconfiguration(msg="Misconfiguration: missing key, 'elasticsearch.index'", expr=None)
	if "type" not in params["elasticsearch"]:
		raise Misconfiguration(msg="Misconfiguration: missing key, 'elasticsearch.type'", expr=None)
	if "cluster_nodes" not in params["elasticsearch"]:
		raise Misconfiguration(msg="Misconfiguration: missing key, 'elasticsearch.cluster_nodes'", expr=None)
	if "path" not in params["logging"]:
		raise Misconfiguration(msg="Misconfiguration: missing key, 'logging.path'", expr=None)
	if "filename" not in params["input"] and "bucket_key" not in params["input"]:
		raise Misconfiguration(msg="Misconfiguration: missing key, 'input.filename' or 'input.bucket_key'", expr=None)
	if "encoding" not in params["input"]:
		raise Misconfiguration(msg="Misconfiguration: missing key, 'input.encoding'", expr=None)

	return True

def write_output_to_file(params, output_list, non_physical, split):
	"""Outputs results to a file"""

	if type(output_list) is queue.Queue:
		output_list = queue_to_list(output_list)

	if len(output_list) < 1:
		logging.warning("No results available to write")
		#return

	# Merge Physical and Non-Physical
	output_list = output_list + non_physical

	# Get File Save Info
	file_path = os.path.basename(split)
	file_name = params["output"]["file"]["processing_location"] + file_path
	file_format = params["output"]["file"].get("format", 'csv')

	# What is the output_list[0]
	#print("Output_list[0]:\n{0}".format(output_list[0]))

	# Get Headers
	header = None
	with open(split, 'r', encoding="utf-8") as infile:
		header = infile.readline()

	# Split on delimiter into a list
	split_header = header.split(params["input"]["delimiter"])
	header_list = [token.strip() for token in split_header]
	header_list.append("IS_PHYSICAL_TRANSACTION")
	header_list.append("z_score_delta")

 	# Get additional fields for display from config file
	additional_fields = params["output"]["results"]["fields"]
	all_fields = header_list + additional_fields
	#print("ALL_FIELDS: {0}".format(all_fields))

	# Output as CSV
	if file_format == "csv":
		delimiter = params["output"]["file"].get("delimiter", ',')
		new_header_list = header_list + params["output"]["results"]["labels"]
		new_header = delimiter.join(new_header_list)
		new_header = new_header.replace("GOOD_DESCRIPTION", "NON_PHYSICAL_TRANSACTION")

		#We only write the header for the first file
		if params["add_header"] is True:
			with open(file_name, 'w', encoding="utf-8") as output_file:
				output_file.write(new_header + "\n")
				params["add_header"] = False

		#We append for every split
		with open(file_name, 'a', encoding="utf-8") as output_file:
			dict_w = csv.DictWriter(output_file, delimiter=delimiter,\
				fieldnames=all_fields, extrasaction='ignore')
			dict_w.writerows(output_list)

	# Output as JSON
	if file_format == "json":
		with open(file_name, 'w') as outfile:
			json.dump(output_list, outfile)

def	hms_to_seconds(t):
	h, m, s = [i.lstrip("0") if len(i.lstrip("0")) != 0 else 0 for i in t.split(':')]
	print ("H: {0}, M: {1}, S: {2}".format(h, m, s))
	return 3600 * float(h) + 60 * float(m) + float(s)

def process_bucket(params):
	"""Process an entire s3 bucket"""

	# Init
	conn = boto.connect_s3()
	s3_location = params["input"]["bucket_key"]
	bucket_name = "s3yodlee"
	regex = "([^/]+)"
	path_regex = re.compile(s3_location + regex)
	bucket = conn.get_bucket(bucket_name, Location.USWest2)
	keep_going = True

	# Loop through Panels in Bucket
	for k in bucket.list():
		if path_regex.search(k.key):

			matches = path_regex.match(k.key)
			input_split_path = params["input"]["split"]["processing_location"]
			new_filename = input_split_path + matches.group(1)

			if keep_going:

				# Download from S3
				keep_going = False
				logging.warning("Fetching %s from S3.", k.key)
				local_input_path = params["input"]["split"]["processing_location"]
				k.get_contents_to_filename(new_filename)
				params["output"]["file"]["name"] = matches.group(1)[:-3]
				logging.warning("Fetch of %s complete.", new_filename)

				# Gunzip
				with gzip.open(new_filename, 'rb') as gzipped_input:
					unzipped_name = new_filename[:-3]
					with open(unzipped_name, 'wb') as unzipped_input:
						for line in gzipped_input:
							unzipped_input.write(line)
						logging.warning("%s unzipped.", new_filename)

				safely_remove_file(new_filename)

				# Run Panel
				logging.warning("Beginning with %s", unzipped_name)
				process_panel(params, unzipped_name, conn)

def process_panel(params, filename, S3):
	"""Process a single panel"""

	row_limit = None
	key = load_hyperparameters(params)
	params["add_header"] = True
	split_count = 0

	# Determine Mode
	if "bank" in filename.lower():
		params["mode"] = "bank"
		classifier = select_model("bank")
	elif "card" in filename.lower():
		params["mode"] = "card"
		classifier = select_model("card")
	else: 
		print("Panel name must include type (bank or card).")

	# Load and Split Files
	try:
		split_path = params["input"]["split"]["processing_location"]
		row_limit = params["input"]["split"]["row_limit"]
		delimiter = params["input"].get("delimiter", "|")
		encoding = params["input"].get("encoding", "utf-8")
		#Break the input file into segments
		split_list = split_csv(open(filename, 'r', encoding=encoding), delimiter=delimiter,
			row_limit=row_limit, output_path=split_path)
	except IOError:
		logging.critical("Invalid ['input']['filename'] key; Input file: %s"
			" cannot be found. Correct your config file.", filename)
		sys.exit()

	# Start a timer
	start_time = datetime.datetime.now()
	split_total = len(split_list)

	logging.warning("There are %i splits to process", split_total)

	# Loop through input segements
	for split in split_list:
		split_count += 1
		logging.warning("Working with the following split: %s", split)
		split_start_time = datetime.datetime.now()
		desc_queue, non_physical = get_desc_queue(split, params, classifier)
		run_meerkat(params, desc_queue, key, non_physical, split)
		end_time = datetime.datetime.now()
		total_time = end_time - start_time
		split_time = end_time - split_start_time
		remaining_splits = split_total - split_count
		completion_percentage = split_count / split_total * 100.0
		my_rate = row_limit / hms_to_seconds(str(split_time))
		logging.warning("Elapsed time: %s, ETA: %s",\
			str(total_time)[:7], str(split_time * remaining_splits)[:7])
		logging.warning("Rate: %10.2f, Completion: %2.2f%%",\
			my_rate, completion_percentage)
		logging.warning("Deleting {0}".format(split))
		safely_remove_file(split)

	# Merge Files, GZIP and Push to S3
	output_location = merge_split_files(params, split_list)
	if params["input"].get("bucket_key", "") != "":
		move_to_S3(params, output_location, S3)

	logging.warning("Complete.")

	# Only do one
	sys.exit()

def mode_switch(params):
	"""Switches mode between, single file / s3 bucket mode"""

	input_file = params["input"].get("filename", "")
	input_bucket = params["input"].get("bucket_key", "")

	if os.path.isfile(input_file):
		print("Processing Single Local File: ", input_file)
		params["output"]["file"]["name"] = os.path.basename(input_file)
		conn = connect_to_S3()
		process_panel(params, input_file, conn)
	elif input_bucket != "":
		print("Processing S3 Bucket: ", input_bucket)
		process_bucket(params)
	else:
		logging.critical("Please provide a local file or s3 bucket for procesing. Terminating")
		sys.exit()

def connect_to_S3():
	"""Returns a connection to S3"""

	try:
		conn = boto.connect_s3()
	except boto.s3.connection.HostRequiredError:
		print("Error connecting to S3, check your credentials")
		sys.exit()

	return conn

def move_to_S3(params, filepath, S3):
	"""Pushes a file back to S3"""

	# Get Connection
	s3_location = "meerkat/output/gpanel/" + params["mode"] + "/"
	key = s3_location + os.path.basename(filepath)
	bucket_name = "s3yodlee"
	bucket = conn.get_bucket(bucket_name, Location.USWest2)

	# Move to S3
	key = bucket.new_key(key)
	key.set_contents_from_filename(filepath)

def run_from_command_line(command_line_arguments):
	"""Runs these commands if the module is invoked from the command line"""
	
	params = initialize()
	validate_params(params)
	mode_switch(params)

if __name__ == "__main__":
	run_from_command_line(sys.argv)
