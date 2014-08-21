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
import string
import io

import boto
import pandas as pd
from boto.s3.connection import Location, S3Connection

from meerkat.custom_exceptions import InvalidArguments, Misconfiguration
from meerkat.consumer import Consumer
from meerkat.binary_classifier.load import select_model
from meerkat.various_tools import load_dict_list, safely_remove_file, load_hyperparameters, safe_print
from meerkat.various_tools import split_csv, merge_split_files, queue_to_list, string_cleanse
from meerkat.various_tools import get_panel_header, get_column_map, get_new_columns
from meerkat.accuracy import test_accuracy, print_results, speed_tests
from meerkat.optimization import run_meerkat as test_meerkat
from meerkat.optimization import get_desc_queue as get_simple_queue
from meerkat.optimization import load_dataset

def get_desc_queue(filename, params, classifier):
	"""Opens a file of descriptions, one per line, and load a description
	queue."""

	encoding = None
	physical, non_physical, atm = [], [], []
	training_fields = ["PHYSICAL_MERCHANT",	"STORE_NUMBER", "CITY",	"STATE", "LATITUDE", "LONGITUDE", "FACTUAL_ID", "STREET"]
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
		safe_print(str(description) + ": " + str(prediction))

		if prediction == "1":
			physical.append(transaction)
		elif prediction == "0":
			non_physical.append(transaction)
			logging.info("NON-PHYSICAL: %s", description)
		elif prediction == "2":
			physical.append(transaction)
			atm.append(transaction)

	# Hold on to GOOD_DESCRIPTION, clear fields
	for row in physical:
		gd = string_cleanse(row["GOOD_DESCRIPTION"])
		row["GOOD_DESCRIPTION"] = string.capwords(gd, " ")

	if params.get("mode", "") == "train":
		for row in physical:
			for field in training_fields:
				row[field] = ""
			row["MERCHANT_NAME"] = row["GOOD_DESCRIPTION"]

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

def run_meerkat(params, desc_queue, hyperparameters, non_physical, split):
	"""Opens a number of threads to process the descriptions queue."""

	# Run the Classifier
	consumer_threads = params.get("concurrency", 8)
	result_queue = queue.Queue()

	for i in range(consumer_threads):
		new_consumer = Consumer(i, params, desc_queue,\
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

	# Shutdown Loggers
	logging.shutdown()

	# Test Accuracy
	if params.get("mode", "") == "test":
		accuracy_results = test_accuracy(params, result_list=result_list, non_physical_trans=non_physical)
		print_results(accuracy_results)
		return accuracy_results

def usage():
	"""Shows the user which parameters to send into the program."""
	result = "Usage:\n\t<path_to_json_format_config_file>"
	logging.error(result)
	return result

def validate_params(params):
	"""Ensures that the correct parameters are supplied."""

	mandatory_keys = ["elasticsearch", "concurrency", "input", "logging", "mode"]

	# Ensure Mandatory Keys are in Config
	for key in mandatory_keys:
		if key not in params:
			raise Misconfiguration(msg="Misconfiguration: missing key, '" + key + "'", expr=None)

	if params["concurrency"] <= 0:
		raise Misconfiguration(msg="Misconfiguration: 'concurrency' must be a positive integer", expr=None)

	if "hyperparameters" not in params["input"]:
		params["input"]["hyperparameters"] = "config/hyperparameters/default.json"

	# Ensure Test Mode Requirements
	if params["mode"] == "test":
		if params.get("verification_source", "") == "":
			raise Misconfiguration(msg="Please provide verification_source to run in test mode", expr=None)

	# Ensure Other Various Parameters Available
	if "index" not in params["elasticsearch"]:
		raise Misconfiguration(msg="Misconfiguration: missing key, 'elasticsearch.index'", expr=None)
	if "type" not in params["elasticsearch"]:
		raise Misconfiguration(msg="Misconfiguration: missing key, 'elasticsearch.type'", expr=None)
	if "cluster_nodes" not in params["elasticsearch"]:
		raise Misconfiguration(msg="Misconfiguration: missing key, 'elasticsearch.cluster_nodes'", expr=None)
	if "path" not in params["logging"]:
		raise Misconfiguration(msg="Misconfiguration: missing key, 'logging.path'", expr=None)
	if "filename" not in params["input"] and "S3" not in params["input"]:
		raise Misconfiguration(msg="Misconfiguration: missing key, 'input.filename' or 'input.S3'", expr=None)
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
	header_list.append("MERCHANT_NAME")

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

def get_S3_buckets(S3_params, conn):
	"""Gets src, dst, and error S3 buckets"""

	bucket_filter = S3_params["filter"]
	src_bucket_name = S3_params["src_bucket_name"]
	src_bucket = conn.get_bucket(src_bucket_name, Location.USWest2)
	dst_bucket_name = S3_params["dst_bucket_name"]
	dst_bucket = conn.get_bucket(dst_bucket_name, Location.USWest2)
	error_bucket_name = S3_params["error_bucket_name"]
	error_bucket = conn.get_bucket(error_bucket_name, Location.USWest2)

	return src_bucket, dst_bucket, error_bucket

def get_S3_regex(S3_params, container, folders):
	"""Get a regex for dealing with S3 buckets"""

	return re.compile(folders + container + "/(" + S3_params["filter"] + "[^/]+)")

def get_completed_files(bucket, path_regex):
	"""Get a list of files that have been processed"""

	completed = {}

	for j in bucket.list():
		if path_regex.search(j.key):
			completed[path_regex.search(j.key).group(1)] = j.size

	return completed

def get_pending_files(bucket, path_regex, completed):
	"""Get a list of files that need to be processed"""

	pending = []

	for k in bucket.list():
		if path_regex.search(k.key):
			file_name = path_regex.search(k.key).group(1)
			if file_name in completed:
				ratio = float(k.size) / completed[file_name]
				if ratio >= 1.8:
					print("Completed Size, Source Size, Ratio: {0}, {1}, {2:.2f}".format(completed[file_name], k.size, ratio))
					print("Re-running {0}".format(file_name))
					pending.append(k)
			elif k.size > 0:
				pending.append(k)

	return pending, completed

def identify_container(params):
	"""Determines whether transactions are bank or card"""

	container = params["container"]
	filename = params["input"]["filename"]

	if "bank" in filename.lower():
		return "bank"
	elif "card" in filename.lower():
		return "card"
	elif container == "bank" or container == "card":
		return container
	else:
		print('Please designate whether this is bank or card in params["container"]')
		sys.exit()

def production_run(params):
	"""Runs Meerkat in production mode"""

	conn = connect_to_S3()
	container = params["container"]
	S3_params =  params["input"]["S3"]

	# Get Buckets
	src_bucket, dst_bucket, error_bucket = get_S3_buckets(S3_params, conn)
	src_s3_path_regex = get_S3_regex(S3_params, container, S3_params["src_folders"])
	dst_s3_path_regex = get_S3_regex(S3_params, container, S3_params["dst_folders"])

	# Sort By Completion Status
	completed = get_completed_files(dst_bucket, dst_s3_path_regex)
	pending, completed = get_pending_files(src_bucket, src_s3_path_regex, completed)

	# Exit if Nothing to Process
	if not pending:
		print("Everything is up-to-date.")
		sys.exit()

	# Process in Reverse Chronological Order
	pending.reverse()

	# Process Files
	for item in pending:

		src_file_name = src_s3_path_regex.search(item.key).group(1)
		dst_file_name = src_file_name

		# Copy from S3
		#item.get_contents_to_filename(S3_params["src_local_path"] + src_file_name)
		params["input"]["filename"] = S3_params["src_local_path"] + src_file_name

		# Load into Dataframe
		container = identify_container(params)
		params["container"] = container
		reader = load_dataframe(params)

		# Process With Meerkat
		run_panel(params, reader)

		# Push to S3

def run_panel(params, reader):
	"""Process a single panel"""

	container = params["container"]
	column_remap = get_column_map(container)
	header = get_panel_header(container)
	new_columns = get_new_columns()

	for df in reader:

		# Rename and add columns
		df = df.rename(columns=column_remap)
		for column in new_columns:
			df[column] = ""

		# Reorder header
		df = df[header]

		# Sort by user
		df = df.sort("UNIQUE_MEM_ID")

		reader.chunksize = reader.chunksize + 100000

		print(df.shape)

	sys.exit()

def load_dataframe(params):
	"""Loads file into a pandas dataframe"""

	params["input"]["filename"] = "/mnt/ephemeral/input/20140109_GPANEL_BANK.txt.gz"

	# Read file into dataframe
	reader = pd.read_csv(params["input"]["filename"], chunksize=100000, compression="gzip", encoding="utf-8", sep='|', error_bad_lines=False)

	return reader

def process_panel(params, filename):
	"""Process a single panel"""

	row_limit = None
	key = load_hyperparameters(params)
	params["add_header"] = True
	split_count = 0
	container = identify_container(params)
	params["container"] = container
	classifier = select_model(container)

	# Load and Split Files
	try:
		split_path = params["input"]["split"]["processing_location"]
		row_limit = params["input"]["split"]["row_limit"]
		delimiter = params["input"].get("delimiter", "|")
		encoding = params["input"].get("encoding", "utf-8")
		#Break the input file into segments
		split_list = split_csv(open(filename, 'r', encoding=encoding, errors='replace'), delimiter=delimiter,
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
	#output_location = merge_split_files(params, split_list)
	#if params["input"].get("bucket_key", "") != "":
	#	move_to_S3(params, output_location, S3)

	#logging.warning("Complete.")

	# Only do one
	print("Exiting after one panel file. Modify code to enable bucket processing.")
	sys.exit()

def mode_switch(params):
	"""Switches mode between, single file / s3 bucket mode"""

	input_file = params["input"].get("filename", "")

	if params.get("mode", "") == "test":
		test_training_data(params)
	elif params.get("mode", "") == "production":
		production_run(params)
	elif os.path.isfile(input_file):
		print("Processing Single Local File: ", input_file)
		params["output"]["file"]["name"] = os.path.basename(input_file)
		conn = connect_to_S3()
		process_panel(params, input_file, conn)
	else:
		logging.critical("Please provide a local file or s3 bucket for procesing. Terminating")
		sys.exit()

def test_training_data(params):
	"""An easy way to test the accuracy of a small set
	provided a set of hyperparameters"""

	hyperparameters = load_hyperparameters(params)
	dataset = load_dataset(params)
	desc_queue = get_simple_queue(dataset)
	test_meerkat(params, desc_queue, hyperparameters)

def connect_to_S3():
	"""Returns a connection to S3"""

	try:
		conn = boto.connect_s3()
	except boto.s3.connection.HostRequiredError:
		print("Error connecting to S3, check your credentials")
		sys.exit()

	return conn

def run_from_command_line(command_line_arguments):
	"""Runs these commands if the module is invoked from the command line"""
	
	params = initialize()
	validate_params(params)
	mode_switch(params)

if __name__ == "__main__":
	run_from_command_line(sys.argv)
