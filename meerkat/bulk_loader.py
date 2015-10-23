#!/usr/local/bin/python3.3

"""The bulk loader is a tool that takes as an input a text based
multifield merchant dataset and indexes it into Elasticsearch for further
processing and analysis against transactions or other merchant datasets

Created on Feb 18, 2014
@author: J. Andrew Key
@author: Matthew Sevrens

Updated on July 17, 2015
@author: Sivan Mehta
"""

#################### USAGE ##########################

# Note: Experts only! Do not touch!

# python3 -m meerkat.bulk_loader config/factual_loader.json

# Estimated runtime: ~40 mins / # number of cluster nodes

#####################################################

import json
import logging
import queue
import re
import sys
import threading

from datetime import datetime
from elasticsearch import Elasticsearch, helpers

from meerkat.custom_exceptions import InvalidArguments, Misconfiguration

def initialize():
	"""Validates the command line arguments."""
	input_file, params = None, None

	if len(sys.argv) != 3:
		#usage()
		raise InvalidArguments(msg="Supply a single argument for the json"\
		+ "formatted configuration file.", expr=None)

	try:
		input_file = open(sys.argv[1], encoding='utf-8')
		params = json.loads(input_file.read())
		input_file.close()
	except IOError:
		logging.error(sys.argv[1] + " not found, aborting.")
		sys.exit()

	if validate_params(params):
		logging.warning("Parameters are valid, proceeding.")

	return params

def load_document_queue(params):
	"""Opens a file of merchants, one per line, and loads a document queue."""

	filename, encoding = None, None
	document_queue = queue.Queue()

	try:
		#filename = params["input"]["filename"]
		filename = sys.argv[2]
		encoding = params["input"]["encoding"]
		with open(filename, 'r', encoding=encoding) as inputfile:
			records = [line.rstrip('\n') for line in inputfile]
	except IOError:
		msg = "Invalid ['input']['filename'] key; Input file: " + filename \
		+ " cannot be found. Correct your config file."
		logging.critical(msg)
		sys.exit()

	header = records.pop(0).split("\t")

	for input_string in records:
		document_queue.put(input_string)
	document_queue_populated = True

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
		if params["logging"]["console"] is True:
			console_handler = logging.StreamHandler()
			console_handler.setLevel(my_level)
			console_handler.setFormatter(my_formatter)
			my_logger.addHandler(console_handler)
		my_logger.info("Log initialized.")

	def run(self):
		"""Eats from the document queue and bulk loads."""
		my_logger = logging.getLogger("thread " + str(self.thread_id))
		params, document_queue = self.params, self.document_queue
		batch_list = self.batch_list
		concurrency_queue = params["concurrency_queue"]
		document_queue_populated = params["document_queue_populated"]
		while True:
			count = 0
			if document_queue_populated and document_queue.empty():
				concurrency_queue.get()
				concurrency_queue.task_done()
				my_logger.info("Consumer finished, concurrency queue size: %i",
					concurrency_queue.qsize())
				return
			for i in range(params["batch_size"]):
				try:
					batch_list.append(document_queue.get(block=False))
					document_queue.task_done()
					count = count + 1
				except queue.Empty:
					if count > 0:
						my_logger.info("queue was empty")
						my_logger.debug("value of i is %i", i)
						self.__publish_batch()
						count = 0
			if count > 0:
				self.__publish_batch()

	def __create_dispersed_fields(self, document):
		"""Routine to create dispersed_fields"""
		dispersed_fields = self.params["elasticsearch"]["dispersed_fields"]
		for field in dispersed_fields:
			keys = field.keys()
			for key in keys:
				my_dict = field[key]
				dispersion_regex = my_dict["dispersion_regex"]
				dispersion_re = re.compile(dispersion_regex)
				components = my_dict['components']
				if "dispersed" not in document:
					document["dispersed"] = {}
				document["dispersed"][key] = {}
				for component in components:
					component_name = component["name"]
					if key in document:
						input_string = document[key]
						if dispersion_re.search(input_string):
							matches = dispersion_re.match(input_string)
							document["dispersed"][key][component_name] =\
								matches.group(component_name)

	def __create_composite_fields(self, document):
		"""Routine to create composite fields"""
		composite_fields = self.params["elasticsearch"]["composite_fields"]
		my_logger = logging.getLogger("thread " + str(self.thread_id))
		for field in composite_fields:
			keys = field.keys()
			for key in keys:
				my_dict = field[key]
				components = my_dict["components"]
				composite_format = my_dict["format"]
				component_values = []
				for component in components:
					if component in document:
						component_values.append(document[component])
					else:
						component_values.append('')
				composite_string = composite_format.format(*component_values).strip()
				if "composite" not in document:
					document["composite"] = {}
				document["composite"][key] = composite_string
				my_logger.debug("New dict is: %s", str(document))

	def __publish_batch(self):
		"""Publishes a bulk index to ElasticSearch."""
		my_logger = logging.getLogger("thread " + str(self.thread_id))
		header = self.params["header"]
		queue_size = len(self.batch_list)
		my_logger.debug("Queue size %i", queue_size)

		#Split the batch into cells built of keys and values, excluding
		#blank values
		params = self.params
		docs, actions = [], []
		len_header = len(header)
		while len(self.batch_list) > 0:
			current_item = self.batch_list.pop(0)
			item_list = current_item.split("\t")
			#If we are missing items at the end, we add blank values
			len_list = len(item_list)
			item_delta = len_header - len_list
			if item_delta > 0:
				item_list.extend([""] * item_delta)
			if len(header) == len(item_list):
				document = {x: y for (x, y) in list(zip(header, item_list)) if y != ""}
				#merge latitude and longitude into a point
				if "longitude" in document and "latitude" in document:
					if document["longitude"].strip() != "" and document["latitude"].strip() != "":
						document["pin"] = {"location":{"type": "point",
							"coordinates" :[ document["longitude"],
							 document["latitude"]]}}
						#my_logger.info(document["pin"])
					del document["longitude"]
					del document["latitude"]
				#If the region is not found, add "XX" instead
				if "region" not in document or document["region"].strip() == "":
					my_logger.critical("Region not found")
					document["region"] = "XX"

				#Routine to create composite fields
				self.__create_composite_fields(document)

				#Routine to create dispersed_fields
				self.__create_dispersed_fields(document)

				docs.append(document)
				action = {
					"_index": params["elasticsearch"]["index"],
					"_type": params["elasticsearch"]["type"],
					"_id": document["factual_id"],
					"_source": document,
					"timestamp": datetime.now()
				}
				actions.append(action)
			else:
				my_logger.info("Header has {0} values, but item has {1} values."(format(len_header, len_item)))
		# Make Calls to Elastic Search
		my_logger.info("Docs: {0}".format(len(docs)))
		success, errors = helpers.bulk(self.es_connection, actions)
		my_logger.info("Success: {0} - Errors: {1}".format(success, errors))
		#_, errors = helpers.bulk(self.es_connection, actions)
		# Evaluate Success Rate
		#success, failure, total = 0, 0, 0
		#for item in errors:
		#	if item["index"]["ok"]:
		#		success += 1
		#	else:
		#		failure += 1
		#	total += 1
		#my_logger.info("Success/Failure/Total: %i/%i/%i - %i documents in queue."
		#	, success, failure, total, self.document_queue.qsize())
		my_logger.info("%i documents in queue.", self.document_queue.qsize())

def start_consumers(params):
	"""Starts our consumer threads"""
	for i in range(params["concurrency"]):
		consumer = ThreadConsumer(i, params)
		consumer.setDaemon(True)
		consumer.start()

def ensure_keys_in_dictionary(dictionary, keys, prefix=''):
	"""Function to determine if the supplied dictionary has the necessary keys."""
	for key in keys:
		if key not in dictionary:
			raise Misconfiguration(msg="Missing key, '" + prefix + key + "'", expr=None)
	return True

def validate_composite_fields(params, my_props):
	"""Validates composite fields."""
	composite_fields = params["elasticsearch"]["composite_fields"]
	my_keys = ["components", "format", "index", "type"]
	for field in composite_fields:
		keys = field.keys()
		for key in keys:
			my_dict = field[key]
			ensure_keys_in_dictionary(my_dict, my_keys,
				prefix="elasticsearch.composite_fields.")

			components = my_dict["components"]
			for component in components:
				if component not in my_props:
					raise Misconfiguration(msg="Component feature '" + component +\
					"' does not exist and cannot be used to build the '" +\
					key + "' feature.", expr=None)

def validate_dispersed_fields(params):
	"""Validates dispersed fields."""
	#Add this:  Validate "dispersed_fields" for good regex, valid components
	dispersed_fields = params["elasticsearch"]["dispersed_fields"]
	my_keys = ["dispersion_regex", "components"]
	for field in dispersed_fields:
		keys = field.keys()
		for key in keys:
			my_dict = field[key]
			ensure_keys_in_dictionary(my_dict, my_keys,
				prefix="elasticsearch.dispersed_fields.")

def validate_boost_vectors(params, my_type, boost_vectors):
	"""Validates boost vectors."""
	my_props = params["elasticsearch"]["type_mapping"]\
		["mappings"][my_type]["properties"]
	missing_keys = dict.fromkeys(boost_vectors.keys() - my_props, "")
	found_keys = []
	for key in missing_keys:
		key_expand = key.replace(".", ".properties.")
		key_split = key_expand.split(".")
		my_dict, key_count, key_total = my_props, 0, len(key_split)
		for second_key in key_split:
			if second_key in my_dict:
				my_dict = my_dict[second_key]
			else:
				raise Misconfiguration(msg="The following boost_vector key " +\
				"is missing from the type mapping: " + str(key), expr=None)
			key_count += 1
			if key_count == key_total:
				logging.warning("key: '" + key + "' was found.")
				found_keys.append(key)
	for key in found_keys:
		del missing_keys[key]
	return missing_keys

def validate_params(params):
	"""Ensures that the correct parameters are supplied."""
	my_keys = ["elasticsearch", "concurrency", "input", "logging", "batch_size"]
	ensure_keys_in_dictionary(params, my_keys)

	my_keys = ["index", "type_mapping", "type", "cluster_nodes",
		"boost_labels", "boost_vectors", "composite_fields", "dispersed_fields"]
	ensure_keys_in_dictionary(params["elasticsearch"], my_keys,
		prefix="elasticsearch.")

	my_keys = ["filename", "encoding"]
	ensure_keys_in_dictionary(params["input"], my_keys, prefix="input.")

	my_keys = ["path", "level", "formatter", "console"]
	ensure_keys_in_dictionary(params["logging"], my_keys, prefix="logging.")

	if params["concurrency"] <= 0:
		raise Misconfiguration(msg="'concurrency' must be a "\
			+ "positive integer", expr=None)

	my_type = params["elasticsearch"]["type"]
	my_props = params["elasticsearch"]["type_mapping"]["mappings"]\
		[my_type]["properties"]

	validate_composite_fields(params, my_props)
	validate_dispersed_fields(params)

	#Ensure that "boost_labels" and "boost_vectors" row vectors have
	#the same cardinality
	label_length = len(params["elasticsearch"]["boost_labels"])
	boost_vectors = params["elasticsearch"]["boost_vectors"]
	for key in boost_vectors:
		if len(boost_vectors[key]) != label_length:
			raise Misconfiguration(msg="Row vector "\
			+ "'elasticsearch.boost_vectors." + key\
			+ "' should have exactly " + str(label_length) + " values.", expr=None)

	#Ensure that "boost_vectors" is a subset of your mapping properties
	add_composite_type_mappings(params)
	add_dispersed_type_mappings(params)

	missing_keys = validate_boost_vectors(params, my_type, boost_vectors)

	if missing_keys:
		raise Misconfiguration(msg="The following boost_vector keys are" +\
		"missing from the type mapping: " + str(missing_keys), expr=None)
	return True

def add_composite_type_mappings(params):
	"""Add additional type mapping properties for composite fields."""
	my_type = params["elasticsearch"]["type"]
	my_properties = params["elasticsearch"]["type_mapping"]["mappings"]\
		[my_type]["properties"]
	composite_fields = params["elasticsearch"]["composite_fields"]
	for field in composite_fields:
		keys = field.keys()
		for key in keys:
			if "composite" not in my_properties:
				my_properties["composite"] = {"properties" : {}}
			my_properties["composite"]["properties"][key] = {
				"index" : field[key]["index"],
				"type" : field[key]["type"]
			}
	logging.critical(json.dumps(params, sort_keys=True, indent=4,
		separators=(',', ':')))

def add_dispersed_type_mappings(params):
	"""Add additional type mapping properties for dispersed fields."""
	my_type = params["elasticsearch"]["type"]
	my_properties = params["elasticsearch"]["type_mapping"]\
		["mappings"][my_type]["properties"]
	dispersed_fields = params["elasticsearch"]["dispersed_fields"]
	if "dispersed" not in my_properties:
		my_properties["dispersed"] = {"properties" : {}}
	for field in dispersed_fields:
		keys = field.keys()
		for key in keys:
			components = field[key]["components"]
			for component in components:
				my_properties["dispersed"]["properties"][component["name"]] = {
					"index" : component["index"],
					"type" : component["type"]
				}
	logging.critical(json.dumps(params, sort_keys=True,
		indent=4, separators=(',', ':')))

def guarantee_index_and_doc_type(params):
	"""Ensure that the index and document type mapping are as they should be"""
	cluster_nodes = params["elasticsearch"]["cluster_nodes"]
	es_index = params["elasticsearch"]["index"]
	doc_type = params["elasticsearch"]["type"]
	es_connection = Elasticsearch(cluster_nodes, sniff_on_start=True,
		sniff_on_connection_fail=True, sniffer_timeout=5, sniff_timeout=5)
	if es_connection.indices.exists(index=es_index):
		logging.critical("Index exists, continuing")
		if es_connection.indices.exists_type(index=es_index, doc_type=doc_type):
			logging.critical("Doc type exists, continuing")
	else:
		logging.warning("Index does not exist, creating")
		index_body = params["elasticsearch"]["type_mapping"]
		result = es_connection.indices.create(index=es_index, body=index_body)
		# okay, acknowledged = result["ok"], result["acknowledged"]
		# try:
		# 	result["ok"] and result["acknowledged"]
		# 	logging.critical("Index created successfully.")
		# except KeyError:
		# 	logging.error("Failed to create index, aborting.")
		# 	sys.exit()

def build_user_index():
	"""Build out the user index"""

	# Get proper params
	try:
		input_file = open("config/user_index.json", encoding='utf-8')
		params = json.loads(input_file.read())
		input_file.close()
	except IOError:
		logging.info("config file for user_index not found." + \
		" Continuing on to factual index")
		return

	# Initialize
	cluster_nodes = params["elasticsearch"]["cluster_nodes"]
	index_name = "user_index"
	es_connection = Elasticsearch(cluster_nodes, sniff_on_start=True,
		sniff_on_connection_fail=True, sniffer_timeout=5, sniff_timeout=5)

	# Build the Index
	if es_connection.indices.exists(index=index_name):
		logging.info("User Index already exists, continuing")
		return
	else:
		logging.warning("User Index does not exist, creating")
		index_body = params["elasticsearch"]["type_mapping"]
		result = es_connection.indices.create(index=index_name, body=index_body)
		okay, acknowledged = result["ok"], result["acknowledged"]
		if okay and acknowledged:
			logging.info("User Index created successfully.")
		else:
			logging.error("Failed to create user index, aborting.")
			sys.exit()

def run_from_command_line():
	"""Runs these commands if the module is invoked from the command line"""

	my_params = initialize()
	logging.warning(json.dumps(my_params, sort_keys=True,
		indent=4, separators=(',', ':')))
	guarantee_index_and_doc_type(my_params)
	my_params["document_queue_populated"] = False
	my_params["concurrency_queue"] = queue.Queue()
	my_params["header"], my_params["document_queue"],\
	my_params["document_queue_populated"] = load_document_queue(my_params)
	start_consumers(my_params)
	my_params["concurrency_queue"].join()
	logging.info("Concurrency joined")
	my_params["document_queue"].join()
	logging.info("Documents joined")

	logging.critical("End of program.")

if __name__ == "__main__":
	run_from_command_line()
