#!/usr/local/bin/python3
# pylint: disable=C0301

"""This bulk loader handles multiple threads."""
import json
import logging
import queue
import sys
import threading

from elasticsearch import Elasticsearch, helpers
from datetime import datetime

from longtail.custom_exceptions import InvalidArguments, Misconfiguration
def initialize():
	"""Validates the command line arguments."""
	input_file, params = None, None

	if len(sys.argv) != 2:
		#usage()
		raise InvalidArguments(msg="Supply a single argument for the json formatted"\
		+ " configuration file.", expr=None)

	try:
		input_file = open(sys.argv[1], encoding='utf-8')
		params = json.loads(input_file.read())
		input_file.close()
	except IOError:
		logging.error(sys.argv[1] + " not found, aborting.")
		sys.exit()

	#params["search_cache"] = {}

	if validate_params(params):
		logging.warning("Parameters are valid, proceeding.")
	return params

def load_document_queue(params):
	"""Opens a file of merchants, one per line, and loads a document queue."""

	lines, filename, encoding = None, None, None
	document_queue = queue.Queue()

	try:
		filename = params["input"]["filename"]
		encoding = params["input"]["encoding"]
		with open(filename, 'r', encoding=encoding) as inputfile:
			lines = inputfile.read()
	except IOError:
		msg = "Invalid ['input']['filename'] key; Input file: " + filename \
		+ " cannot be found. Correct your config file."
		logging.critical(msg)
		sys.exit()

	records = lines.split("\n")
	header = records.pop(0).split("\t")
	for input_string in records:
		document_queue.put(input_string)
	#document_queue.task_done()
	document_queue_populated = True

	logging.warning("Document Queue has " + str(document_queue.qsize()) + " elements")
	return header, document_queue, document_queue_populated

class ThreadConsumer(threading.Thread):
	"""Used for our consumer threads."""
	def __init__(self, thread_id, params):
		"""Constructor"""
		threading.Thread.__init__(self)
		self.thread_id = thread_id
		self.params = params
		self.document_queue = params["document_queue"]
		self.params["concurrency_queue"].put(self.thread_id)
		cluster_nodes = self.params["elasticsearch"]["cluster_nodes"]
		self.es_connection = Elasticsearch(cluster_nodes, sniff_on_start=True,
			sniff_on_connection_fail=True, sniffer_timeout=5, sniff_timeout=5)
		self.__set_logger()
		self.batch_list = []

	def __set_logger(self):
		"""Creates a logger, based upon the supplied config object."""
		levels = {'debug': logging.DEBUG, 'info': logging.INFO\
		, 'warning': logging.WARNING, 'error': logging.ERROR\
		, 'critical': logging.CRITICAL}
		params = self.params
		my_level = params["logging"]["level"]
		if my_level in levels:
			my_level = levels[my_level]
		my_path = params["logging"]["path"]
		my_formatter = logging.Formatter(params['logging']['formatter'])
		#You'll want to add something to identify the thread
		my_logger = logging.getLogger("thread " + str(self.thread_id))
		my_logger.setLevel(my_level)
		file_handler = logging.FileHandler(my_path)
		file_handler.setLevel(my_level)
		file_handler.setFormatter(my_formatter)
		my_logger.addHandler(file_handler)

		#Add console logging, if configured
		my_console = params["logging"]["console"]
		if my_console is True:
			console_handler = logging.StreamHandler()
			console_handler.setLevel(my_level)
			console_handler.setFormatter(my_formatter)
			my_logger.addHandler(console_handler)
		my_logger.info("Log initialized.")

	def run(self):
		"""Eats from the document queue and bulk loads."""
		my_logger = logging.getLogger("thread " + str(self.thread_id))
		params, document_queue = self.params, self.document_queue
		batch_list, concurrency_queue = self.batch_list, params["concurrency_queue"]
		document_queue_populated = params["document_queue_populated"]
		while True:
			count = 0
			if document_queue_populated and document_queue.empty():
				my_logger.info("Concurrency queue size: " + str(concurrency_queue.qsize()))
				concurrency_queue.get()
				concurrency_queue.task_done()
				my_logger.info("Consumer finished.")
				cq_size = concurrency_queue.qsize()
				dq_size = document_queue.qsize()
				my_logger.info("Concurrency queue size: " + str(cq_size))
				my_logger.info("Document queue size: " + str(dq_size))
				return
			for i in range(params["batch_size"]):
				try:
					batch_list.append(document_queue.get(block=False))
					document_queue.task_done()
					count = count + 1
				except queue.Empty:
					if count > 0:
						my_logger.info("queue was empty")
						self.__publish_batch()
						count = 0
			if count > 0:
				self.__publish_batch()

	def __publish_batch(self):
		"""You do nothing but log."""
		logger = logging.getLogger("thread " + str(self.thread_id))
		header = self.params["header"]
		queue_size = len(self.batch_list)
		logger.info("Queue size " + str(queue_size))

		#Split the batch into cells built of keys and values, excluding blank values
		params = self.params
		docs, actions = [], []
		while len(self.batch_list) > 0:
			item_list = self.batch_list.pop(0).split("\t")
			if len(header) == len(item_list):
				d = {x: y for (x, y) in list(zip(header, item_list)) if y != ""}
				docs.append(d)
				action = {
					"_index": params["elasticsearch"]["index"],
					"_type": params["elasticsearch"]["type"],
					"_id": d["factual_id"],
					"_source": d,
					"timestamp": datetime.now()
				}
				actions.append(action)
		helpers.bulk(self.es_connection, actions)
		logger.info("Done for " + str(queue_size))

def start_consumers(params):
	"""Starts our consumer threads"""
	for i in range(params["concurrency"]):
		c = ThreadConsumer(i,params)
		c.setDaemon(True)
		c.start()

def start_producers(params):
	"""Starts our producer threads"""
	for i in range(1):
		c = ThreadProducer(i,params)
		c.setDaemon(True)
		c.start()

def validate_params(params):
	"""Ensures that the correct parameters are supplied."""
	mandatory_keys = ["elasticsearch", "concurrency", "input", "logging", "batch_size"]
	for key in mandatory_keys:
		if key not in params:
			raise Misconfiguration(msg="Misconfiguration: missing key, '" + key + "'", expr=None)

	if params["concurrency"] <= 0:
		raise Misconfiguration(msg="Misconfiguration: 'concurrency' must be a positive integer", expr=None)

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

if __name__ == "__main__":
	#Runs the entire program.
	PARAMS = initialize()
	logging.warning(json.dumps(PARAMS, sort_keys=True, indent=4, separators=(',', ':')))
	PARAMS["document_queue_populated"] = False
	PARAMS["concurrency_queue"] = queue.Queue()
	PARAMS["header"], PARAMS["document_queue"], PARAMS["document_queue_populated"] =\
		load_document_queue(PARAMS)
	#start_producers(PARAMS)
	start_consumers(PARAMS)
	PARAMS["concurrency_queue"].join()
	logging.info("Concurrency joined")
	PARAMS["document_queue"].join()
	logging.info("Documents joined")

	logging.critical("End of program.")

