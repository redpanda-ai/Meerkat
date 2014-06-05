#!/usr/local/bin/python3

"""This script scans, tokenizes, and constructs queries to match transaction
description strings (unstructured data) to merchant data indexed with
ElasticSearch (structured data).

@author: J. Andrew Key
@author: Matthew Sevrens
"""

import csv
import datetime
import collections
import json
import logging
import os
import pickle
import queue
import sys

from .custom_exceptions import InvalidArguments, Misconfiguration
from .description_consumer import DescriptionConsumer
from .binary_classifier.load import predict_if_physical_transaction
from .various_tools import load_dict_list, split_csv
from .accuracy import test_accuracy, print_results, speed_tests

def get_desc_queue(filename, params):
	"""Opens a file of descriptions, one per line, and load a description
	queue."""

	encoding = None
	physical, non_physical, atm = [], [], []
	#users = collections.defaultdict(list)
	desc_queue = queue.Queue()

	try:
		encoding = params["input"]["encoding"]
		delimiter = params["input"].get("delimiter", "|")
		transactions = load_dict_list(filename, encoding=encoding,\
			delimiter=delimiter)
	except IOError:
		logging.critical("Invalid ['input']['filename'] key; Input file: %s"
			" cannot be found. Correct your config file.", filename)
		sys.exit()

	# Run Binary Classifier
	for transaction in transactions:
		transaction['factual_id'] = ""
		description = transaction["DESCRIPTION"]
		prediction = predict_if_physical_transaction(description)
		transaction["IS_PHYSICAL_TRANSACTION"] = prediction
		#print(description + ": " + prediction)
		if prediction == "1":
			physical.append(transaction)
		elif prediction == "0":
			non_physical.append(transaction)
			logging.info("NON-PHYSICAL: %s", description)
		elif prediction == "2":
			physical.append(transaction)
			atm.append(transaction)

	# Split into user buckets
	#for row in physical:
	#	user = row.get("UNIQUE_MEM_ID", "")
	#	users[user].append(row)

	# Add Users to Queue
	#for key, _ in users.items():
	#	desc_queue.put(users[key])

	#atm_percent = (len(atm) / len(transactions)) * 100
	#non_physical_percent = (len(non_physical) / len(transactions)) * 100
	#physical_percent = (len(physical) / len(transactions)) * 100

	#print("")
	#print("PHYSICAL: ", round(physical_percent, 2), "%")
	#print("NON-PHYSICAL: ", round(non_physical_percent, 2), "%")
	#print("ATM: ", round(atm_percent, 2), "%")
	#print("")

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

def tokenize(params, desc_queue, hyperparameters, non_physical):
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
		write_output_to_file(params, result_list, non_physical)
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

def write_output_to_file(params, output_list, non_physical):
	"""Outputs results to a file"""

	if type(output_list) is queue.Queue:
		output_list = queue_to_list(output_list)

	if len(output_list) < 1:
		logging.warning("No results available to write")
		#return

	# Merge Physical and Non-Physical
	output_list = output_list + non_physical

	# Get File Save Info
	file_name = params["output"]["file"].get("path",\
		'../data/output/meerkatLabeled.csv')
	file_format = params["output"]["file"].get("format", 'csv')

	# What is the output_list[0]
	#print("Output_list[0]:\n{0}".format(output_list[0]))

	# Get Headers
	header = None
	with open(params["input"]["filename"], 'r') as infile:
		header = infile.readline()

	# Split on delimiter into a list
	split_header = header.split(params["input"]["delimiter"])
	header_list = [token.strip() for token in split_header]

 	# Get additional fields for display from config file
	additional_fields = params["output"]["results"]["fields"]
	all_fields = header_list + additional_fields
	#print("ALL_FIELDS: {0}".format(all_fields))

	# Output as CSV
	if file_format == "csv":
		delimiter = params["output"]["file"].get("delimiter", ',')
		new_header_list = header_list + params["output"]["results"]["labels"]
		new_header = delimiter.join(new_header_list)

		with open(file_name, 'a') as output_file:
			#We only write the header for the first file
			if params["add_header"] is True:
				output_file.write(new_header + "\n")
				params["add_header"] = False
			dict_w = csv.DictWriter(output_file, delimiter=delimiter,\
				fieldnames=all_fields, extrasaction='ignore')
			dict_w.writerows(output_list)

	# Output as JSON
	if file_format == "json":
		with open(file_name, 'w') as outfile:
			json.dump(output_list, outfile)

def begin():
	params = initialize()
	key = load_hyperparameters(params)
	try:
		filename = params["input"]["filename"]
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

	#Loop through input segements
	params["add_header"] = True
	for split in split_list:
		print("Working with the following split: {0}".format(split))
		desc_queue, non_physical = get_desc_queue(filename, params)
		tokenize(params, desc_queue, key, non_physical)
	print("Complete.")

if __name__ == "__main__":
	begin()

