#!/usr/local/bin/python3

"""This script scans, tokenizes, and constructs queries to match transaction
description strings (unstructured data) to merchant data indexed with
ElasticSearch (structured data).

@author: J. Andrew Key
@author: Matthew Sevrens
"""

import boto
import csv
import datetime
import collections
import gzip
import json
import logging
import os
import pickle
import re
import queue
import sys

from .custom_exceptions import InvalidArguments, Misconfiguration
from .description_consumer import DescriptionConsumer
from .binary_classifier.load import predict_if_physical_transaction
from .various_tools import load_dict_list, safely_remove_file, split_csv
from .accuracy import test_accuracy, print_results, speed_tests

from boto.s3.connection import Location, S3Connection

def get_desc_queue(filename, params):
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
		description = transaction["DESCRIPTION"]
		prediction = predict_if_physical_transaction(description)
		transaction["IS_PHYSICAL_TRANSACTION"] = prediction
		print(description + ": " + prediction)
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

	params["search_cache"] = {}

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

def load_pickle_cache(params):
	"""Loads the Pickled Cache"""

	try:
		with open("search_cache.pickle", 'rb') as client_cache_file:
			params["search_cache"] = pickle.load(client_cache_file)
	except IOError:
		logging.critical("search_cache.pickle not found, starting anyway.")

	if validate_params(params):
		logging.info("Parameters are valid, proceeding.")

	return params

def tokenize(params, desc_queue, hyperparameters, non_physical, split):
	"""Opens a number of threads to process the descriptions queue."""

	# Load Pickle Cache if enough transactions
	use_cache = params["elasticsearch"].get("cache_results", "true")

	if use_cache == "true":
		params = load_pickle_cache(params)

	# Run the Classifier
	consumer_threads = params.get("concurrency", 8)
	result_queue = queue.Queue()
	#start_time = datetime.datetime.now()

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

	# Do Speed Tests
	#speed_tests(start_time, accuracy_results)

	# Save the Cache
	#save_pickle_cache(params)

	# Shutdown Loggers
	logging.shutdown()

	#return accuracy_results

def save_pickle_cache(params):
	"""Saves search results into a picked file."""
	use_cache = params["elasticsearch"].get("cache_results", True)

	if use_cache == False:
		return

	# Destroy the out-dated cache
	logging.critical("Removing original pickle")
	try:
		os.remove("search_cache.pickle")
	except OSError:
		pass

	# Pickle the search_cache
	logging.critical("Begin Pickling.")
	with open('search_cache.pickle', 'wb') as client_cache_file:
		pickle.dump(params["search_cache"], client_cache_file, pickle.HIGHEST_PROTOCOL)
	logging.critical("Pickling complete.")

def usage():
	"""Shows the user which parameters to send into the program."""
	result = "Usage:\n\t<path_to_json_format_config_file>"
	logging.error(result)
	return result

def validate_params(params):
	"""Ensures that the correct parameters are supplied."""
	mandatory_keys = ["elasticsearch", "concurrency", "input", "logging"]
	for key in mandatory_keys:
		if key not in params:
			raise Misconfiguration(msg="Misconfiguration: missing key, '" + key + "'", expr=None)

	if params["concurrency"] <= 0:
		raise Misconfiguration(msg="Misconfiguration: 'concurrency' must be a positive integer", expr=None)

	if "hyperparameters" not in params["input"]:
		params["input"]["hyperparameters"] = "config/hyperparameters/default.json"

	if "index" not in params["elasticsearch"]:
		raise Misconfiguration(msg="Misconfiguration: missing key, 'elasticsearch.index'", expr=None)
	if "type" not in params["elasticsearch"]:
		raise Misconfiguration(msg="Misconfiguration: missing key, 'elasticsearch.type'", expr=None)
	if "cluster_nodes" not in params["elasticsearch"]:
		raise Misconfiguration(msg="Misconfiguration: missing key, 'elasticsearch.cluster_nodes'", expr=None)
	if "path" not in params["logging"]:
		raise Misconfiguration(msg="Misconfiguration: missing key, 'logging.path'", expr=None)
	if "filename" not in params["input"]:
		raise Misconfiguration(msg="Misconfiguration: missing key, 'input.filename'", expr=None)
	if "encoding" not in params["input"]:
		raise Misconfiguration(msg="Misconfiguration: missing key, 'input.encoding'",\
			expr=None)
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
	file_name = params["output"]["file"]["path"] + file_path
	file_format = params["output"]["file"].get("format", 'csv')

	# What is the output_list[0]
	#print("Output_list[0]:\n{0}".format(output_list[0]))

	# Get Headers
	header = None
	with open(split, 'r') as infile:
		header = infile.readline()

	# Split on delimiter into a list
	split_header = header.split(params["input"]["delimiter"])
	header_list = [token.strip() for token in split_header]
	header_list.append("IS_PHYSICAL_TRANSACTION")

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
			with open(file_name, 'w') as output_file:
				output_file.write(new_header + "\n")
				params["add_header"] = False

		#We append for every split
		with open(file_name, 'a') as output_file:
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

	try:
		conn = boto.connect_s3()
	except boto.s3.connection.HostRequiredError:
		print("Error connecting to S3, check your credentials")

	bucket_string = "s3yodlee"
	path_string = "meerkat/input/gpanel/card/([^/]+)"
	path_regex = re.compile(path_string)
	bucket = conn.get_bucket(bucket_string,Location.USWest2)
	keep_going = True

	for k in bucket.list():
		if path_regex.search(k.key):
			matches = path_regex.match(k.key)
			input_split_path = params["input"]["split"]["path"]
			new_filename = input_split_path + matches.group(1)
			if keep_going:
				keep_going = False
				#print(k.key, k.size, k.encrypted)
				logging.warning("Fetching %s from S3.", k.key)
				#print("Creating new file at {0}.".format(new_filename))
				local_input_path = params["input"]["split"]["path"]
				k.get_contents_to_filename(new_filename)
				params["output"]["file"]["name"] = matches.group(1)[:-3]
				logging.warning("Fetch of %s complete.", new_filename)
				with gzip.open(new_filename, 'rb') as gzipped_input:
					unzipped_name = new_filename[:-3]
					with open(unzipped_name, 'wb') as unzipped_input:
						unzipped_input.write(gzipped_input.read())
						logging.warning("%s unzipped.", new_filename)
				safely_remove_file(new_filename)
				logging.warning("Beginning with %s", unzipped_name)
				process_panel(params, unzipped_name)

def process_panel(params, filename):
	"""Process a single panel"""

	row_limit = None
	key = load_hyperparameters(params)
	try:
		#filename = params["input"]["filename"]
		split_path = params["input"]["split"]["path"]
		row_limit = params["input"]["split"]["row_limit"]
		delimiter = params["input"].get("delimiter", "|")
		#Break the input file into segments
		split_list = split_csv(open(filename, 'r'), delimiter=delimiter,
			row_limit=row_limit, output_path=split_path)
	except IOError:
		logging.critical("Invalid ['input']['filename'] key; Input file: %s"
			" cannot be found. Correct your config file.", filename)
		sys.exit()

	#Removing original file
	safely_remove_file(filename)

	#Loop through input segements
	params["add_header"] = True
	#Start a timer
	split_count = 0
	split_total = len(split_list)
	logging.warning("There are %i splits to process", split_total)
	#print("There are {0} splits to process.".format(split_total))
	start_time = datetime.datetime.now()
	for split in split_list:
		split_count += 1
		logging.warning("Working with the following split: %s", split)
		split_start_time = datetime.datetime.now()
		desc_queue, non_physical = get_desc_queue(split, params)
		tokenize(params, desc_queue, key, non_physical, split)
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
	merge_split_files(params, split_list)
		
	logging.warning("Complete.")

def merge_split_files(params, split_list):
	"""Takes a split list and merges the files back together
	after processing is complete"""

	file_name = params["output"]["file"]["name"]
	base_path = params["output"]["file"]["path"]
	first_file = base_path + split_list.pop(0)
	output = open(file_name, "a")

	for line in open(first_file):
		output.write(line)

	for split in split_list:
		with open(base_path + split) as chunk:
			chunk.next()
			for line in chunk:
				output.write(line)
		safely_remove_file(base_path + split)

	output.close()

if __name__ == "__main__":
	#Runs the entire program.

	params = initialize()
	process_bucket(params)
