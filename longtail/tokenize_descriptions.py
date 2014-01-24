#!/usr/local/bin/python3
# pylint: disable=C0301

"""This script scans, tokenizes, and constructs queries to match transaction
description strings (unstructured data) to merchant data indexed with
ElasticSearch (structured data)."""

import datetime, json, logging, queue, sys, csv
from longtail.custom_exceptions import InvalidArguments
from longtail.description_consumer import DescriptionConsumer

def get_desc_queue(params):
	"""Opens a file of descriptions, one per line, and load a description
	queue."""
	lines = None
	desc_queue = queue.Queue()
	try:
		input_file = open(params["input"]["filename"]\
		, encoding=params['input']['encoding'])
		lines = input_file.read()
	except FileNotFoundError:
		print(sys.argv[1], " not found, aborting.")
		logging.error(sys.argv[1] + " not found, aborting.")
		sys.exit()
	for input_string in lines.split("\n"):
		desc_queue.put(input_string)
	input_file.close()
	return desc_queue

def initialize():
	"""Validates the command line arguments."""
	input_file = None
	if len(sys.argv) != 2:
		usage()
		raise InvalidArguments(msg="Incorrect number of arguments", expr=None)
	try:
		input_file = open(sys.argv[1], encoding='utf-8')
		params = json.loads(input_file.read())
		input_file.close()
	except FileNotFoundError:
		print(sys.argv[1], " not found, aborting.")
		logging.error(sys.argv[1] + " not found, aborting.")
		sys.exit()
	return params

def tokenize(params, desc_queue):
	"""Opens a number of threads to process the descriptions queue."""
	consumer_threads = 1
	result_queue = queue.Queue()
	if "concurrency" in params:
		consumer_threads = params["concurrency"]
	start_time = datetime.datetime.now()
	for i in range(consumer_threads):
		new_consumer = DescriptionConsumer(i, params, desc_queue, result_queue)
		new_consumer.setDaemon(True)
		new_consumer.start()
	desc_queue.join()
	#Writing to an output file, if necessary.
	if "file" in params["output"] and "format" in params["output"]["file"]\
	and params["output"]["file"]["format"] in ["csv", "json"]:
		write_output_to_file(params, result_queue)
	else:
		logging.critical("Not configured for file output.")
	time_delta = datetime.datetime.now() - start_time
	logging.critical("Total Time Taken: " + str(time_delta))

def write_output_to_file(params, result_queue):
	"""Outputs results to a file"""
	output_list = []
	while result_queue.qsize() > 0:
		try:
			output_list.append(result_queue.get())
			result_queue.task_done()

		except queue.Empty:
			break
	result_queue.join()
	file_name = params["output"]["file"].get("path", '../data/longtailLabeled.csv')
	# Output as CSV
	if params["output"]["file"]["format"] == "csv":
		delimiter = params["output"]["file"].get("delimiter", ',')
		output_file = open(file_name, 'w')
		dict_w = csv.DictWriter(output_file, delimiter=delimiter, fieldnames=output_list[0].keys())
		dict_w.writeheader()
		dict_w.writerows(output_list)
		output_file.close()
	# Output as JSON
	if params["output"]["file"]["format"] == "json":
		with open(file_name, 'w') as outfile:
			json.dump(output_list, outfile)

def usage():
	"""Shows the user which parameters to send into the program."""
	result = "Usage:\n\t<quoted_transaction_description_string>"
	print(result)
	return result

if __name__ == "__main__":
	#Runs the entire program.
	PARAMS = initialize()
	DESC_QUEUE = get_desc_queue(PARAMS)
	tokenize(PARAMS, DESC_QUEUE)
	