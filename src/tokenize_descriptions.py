#!/usr/local/bin/python3

"""This script scans, tokenizes, and constructs queries to match transaction
description strings (unstructured data) to merchant data indexed with
ElasticSearch (structured data)."""

import datetime, json, logging, queue, sys
from custom_exceptions import InvalidArguments
from description_consumer import DescriptionConsumer

def get_desc_queue(params):
	"""Opens a file of descriptions, one per line, and load a description
	queue."""
	lines = None
	desc_queue = queue.Queue()
	try:
		input_file = open(params["input"]["filename"]\
		, encoding=params['input']['encoding'])
		lines = input_file.read()
		#input_file.close()
	except FileNotFoundError:
		print (sys.argv[1], " not found, aborting.")
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
		print (sys.argv[1], " not found, aborting.")
		logging.error(sys.argv[1] + " not found, aborting.")
		sys.exit()
	return params

def tokenize(params, desc_queue):
	"""Opens a number of threads to process the descriptions queue."""
	consumer_threads = 1
	if "concurrency" in params:
		consumer_threads = params["concurrency"]
	start_time = datetime.datetime.now()
	for i in range(consumer_threads):
		new_consumer = DescriptionConsumer(i, params, desc_queue)
		new_consumer.setDaemon(True)
		new_consumer.start()
	desc_queue.join()
	time_delta = datetime.datetime.now() - start_time
	logging.critical("Total Time Taken: " + str(time_delta))

def usage():
	"""Shows the user which parameters to send into the program."""
	print( "Usage:\n\t<quoted_transaction_description_string>")

#Test to ensure that changes are uploaded only to the issue_5 branch.
STILL_BREAKABLE = 2
PARAMS = initialize()
DESC_QUEUE = get_desc_queue(PARAMS)
tokenize(PARAMS, DESC_QUEUE)
