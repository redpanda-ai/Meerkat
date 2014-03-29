#!/usr/local/bin/python3
# pylint: disable=C0301

"""This script scans, tokenizes, and constructs queries to match transaction
description strings (unstructured data) to merchant data indexed with
ElasticSearch (structured data).

@author: J. Andrew Key
@author: Matthew Sevrens

"""

import csv
import datetime
import json
import logging
import os
import pickle
import queue
import sys

from pprint import pprint
from operator import itemgetter
from longtail.custom_exceptions import InvalidArguments, Misconfiguration
from longtail.description_consumer import DescriptionConsumer
from longtail.binary_classifier.load import predict_if_physical_transaction
from longtail.various_tools import load_dict_list
from longtail.accuracy import test_accuracy, print_results, speed_tests

def get_desc_queue(params):
	"""Opens a file of descriptions, one per line, and load a description
	queue."""

	lines, filename, encoding = None, None, None
	non_physical = []
	atm = []
	desc_queue = queue.Queue()

	try:
		filename = params["input"]["filename"]
		encoding = params["input"]["encoding"]
		transactions = load_dict_list(filename, encoding=encoding)
	except IOError:
		logging.critical("Invalid ['input']['filename'] key; Input file: %s"
			" cannot be found. Correct your config file.", filename)
		sys.exit()

	# Sort By MEM_ID
	if "MEM_ID" in transactions[0]:
		transactions = sorted(transactions, key=itemgetter('MEM_ID')) 

	# Run Binary Classifier
	for transaction in transactions:
		description = transaction["DESCRIPTION"]
		prediction = predict_if_physical_transaction(description)
		if prediction == "1":
			desc_queue.put(transaction)
		elif prediction == "0":
			non_physical.append(transaction)
			logging.info("NON-PHYSICAL: %s", description)
		elif prediction == "2":
			desc_queue.put(transaction)
			atm.append(transaction)

	atm_percent = (len(atm) / len(transactions)) * 100
	non_physical_percent = (len(non_physical) / len(transactions)) * 100
	physical_percent = (desc_queue.qsize() / len(transactions)) * 100

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
		with open("search_cache.pickle", 'rb') as f:
			params["search_cache"] = pickle.load(f)
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
	start_time = datetime.datetime.now()

	for i in range(consumer_threads):
		new_consumer = DescriptionConsumer(i, params, desc_queue, result_queue, hyperparameters)
		new_consumer.setDaemon(True)
		new_consumer.start()
	desc_queue.join()

	# Convert queue to list
	result_list = queue_to_list(result_queue)

	# Writing to an output file, if necessary.
	if "file" in params["output"] and "format" in params["output"]["file"]\
	and params["output"]["file"]["format"] in ["csv", "json"]:
		write_output_to_file(params, result_list)
	else:
		logging.critical("Not configured for file output.")

	# Test Accuracy
	accuracy_results = test_accuracy(params, result_list=result_list, non_physical_trans=non_physical)
	print_results(accuracy_results)

	# Do Speed Tests
	speed_tests(start_time, accuracy_results)

	# Save the Cache
	save_pickle_cache(params)

	# Shutdown Loggers
	logging.shutdown()

	return accuracy_results

def save_pickle_cache(params):

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
	with open('search_cache.pickle', 'wb') as f:
		pickle.dump(params["search_cache"], f, pickle.HIGHEST_PROTOCOL)
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
		raise Misconfiguration(msg="Misconfiguration: missing key, 'input.encoding'", expr=None)
	return True

def write_output_to_file(params, output_list):
	"""Outputs results to a file"""

	if type(output_list) is queue.Queue:
		output_list = queue_to_list(output_list)

	if len(output_list) < 1:
		logging.warning("No results available to write")
		return

	file_name = params["output"]["file"].get("path", '../data/output/longtailLabeled.csv')
	file_format = params["output"]["file"].get("format", 'csv')

	# Output as CSV
	if file_format == "csv":
		delimiter = params["output"]["file"].get("delimiter", ',')
		output_file = open(file_name, 'w')
		dict_w = csv.DictWriter(output_file, delimiter=delimiter, fieldnames=output_list[0].keys())
		dict_w.writeheader()
		dict_w.writerows(output_list)
		output_file.close()

	# Output as JSON
	if file_format == "json":
		with open(file_name, 'w') as outfile:
			json.dump(output_list, outfile)

if __name__ == "__main__":
	#Runs the entire program.
	PARAMS = initialize()
	KEY = load_hyperparameters(PARAMS)
	DESC_QUEUE, NON_PHYSICAL = get_desc_queue(PARAMS)
	tokenize(PARAMS, DESC_QUEUE, KEY, NON_PHYSICAL)
