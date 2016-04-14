#!/usr/local/bin/python3.3

"""This is where we keep functions that are useful enough to call from
within multiple scripts.

Created on Dec 20, 2013
@author: J. Andrew Key
@author: Matthew Sevrens
"""

import csv
import gzip
import json
import logging
import os
import re
import sys
import queue

import boto
import numpy as np
import pandas as pd

from jsonschema import validate

CLEAN_PATTERN = re.compile(r"\\+\|")
QUOTE_CLEAN = re.compile(r'\"')

def validate_configuration(config, schema):
	"""validate a json config file"""
	try:
		try:
			config = load_params(config)
			schema = load_params(schema)
		except ValueError as val_err:
			logging.error("Config file is mal-formatted {0}".format(val_err))
			sys.exit()
		validate(config, schema)
		logging.warning("Configuration schema is valid.")
	except IOError:
		logging.error("File not found, aborting.")
		sys.exit()
	return config

def load_dict_list(file_name, encoding='utf-8', delimiter="|"):
	"""Loads a dictionary of input from a file into a list."""
	input_file = open(file_name, encoding=encoding, errors='replace')
	dict_list = list(csv.DictReader(input_file, delimiter=delimiter,\
		quoting=csv.QUOTE_NONE))
	input_file.close()
	return dict_list

def load_dict_ordered(file_name, encoding='utf-8', delimiter="|"):
	"""Loads a dictionary of input, anf returns a list and ordered
	fieldnames"""
	input_file = open(file_name, encoding=encoding, errors='replace')
	reader = csv.DictReader(input_file, delimiter=delimiter,\
		quoting=csv.QUOTE_NONE)
	dict_list = list(reader)
	input_file.close()
	return dict_list, reader.fieldnames

def load_piped_dataframe(filename, chunksize=False, usecols=False):
	"""Load piped dataframe from file name"""

	options = {
		"quoting": csv.QUOTE_NONE,
		"na_filter": False,
		"encoding": "utf-8",
		"sep": "|",
		"error_bad_lines": False
	}

	if usecols:
		columns = usecols
		options["usecols"] = usecols
	else:
		with open(filename, 'r') as reader:
			header = reader.readline()
		columns = header.split("|")

	options["dtype"] = {c: "object" for c in columns}

	if isinstance(chunksize, int):
		options["chunksize"] = chunksize

	return pd.read_csv(filename, **options)

def write_dict_list(dict_list, file_name, encoding="utf-8", delimiter="|",\
		column_order=""):
	""" Saves a lists of dicts with uniform keys to file """
	if column_order == "":
		column_order = dict_list[0].keys()

	with open(file_name, 'w', encoding=encoding, errors='replace') as output_file:
		dict_w = csv.DictWriter(output_file, delimiter=delimiter,\
			fieldnames=column_order, extrasaction='ignore')
		dict_w.writeheader()
		dict_w.writerows(dict_list)

def to_stdout(string, errors="replace"):
	"""Converts a string to stdout compatible encoding"""

	encoded = string.encode(sys.stdout.encoding, errors)
	decoded = encoded.decode(sys.stdout.encoding)
	return decoded

def safe_print(*objs, errors="replace"):
	"""Print without unicode errors"""
	print(*(to_stdout(str(o), errors) for o in objs))

def safe_input(prompt=""):
	"""Safely input a string"""

	try:
		return input(prompt)
	except KeyboardInterrupt:
		sys.exit()

	# there are no other exceptions that can occur in this scenario
	return ""

def progress(i, my_list, message=""):
	"""Display progress percent in a loop"""
	my_progress = (i / len(my_list)) * 100
	my_progress = str(round(my_progress, 1)) + "% " + message
	sys.stdout.write('\r')
	sys.stdout.write(my_progress)
	sys.stdout.flush()

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

def load_params(filename):
	"""Load a set of parameters provided a filename"""
	if isinstance(filename, str):
		input_file = open(filename, encoding='utf-8')
		params = json.loads(input_file.read())
		input_file.close()
		return params
	return filename

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

def get_es_connection(params):
	"""Fetch a connection to the factual index"""

	from elasticsearch import Elasticsearch

	cluster_nodes = params["elasticsearch"]["cluster_nodes"]
	index = params["elasticsearch"]["index"]
	es_connection = Elasticsearch(cluster_nodes, index=index,\
		sniff_on_start=False, sniff_on_connection_fail=False,\
		timeout=30)

	return es_connection

def get_s3_connection():
	"""Returns a connection to S3"""
	try:
		conn = boto.connect_s3()
	except boto.s3.connection.HostRequiredError:
		print("Error connecting to S3, check your credentials")
		sys.exit()
	return conn

def post_sns(message):
	"""Post an SNS message"""
	region = 'us-west-2'
	topic = 'arn:aws:sns:us-west-2:003144629351:panel_6m'
	conn = boto.sns.connect_to_region(region)
	_ = conn.publish(topic=topic, message=message)

def get_merchant_by_id(*args, **kwargs):
	"""Fetch the details for a single factual_id"""
	params, factual_id, es_connection = args[:]
	index = kwargs.get('index', "")
	doc_type = kwargs.get('doc_type', "factual_type")
	routing = kwargs.get('routing', None)

	if index == "":
		index = params.get("elasticsearch", {}).get("index", "")

	if factual_id == "NULL":
		return None

	try:
		if routing:
			result = es_connection.get(index=index, doc_type=doc_type, id=factual_id, routing=routing)
		else:
			result = es_connection.get(index=index, doc_type=doc_type, id=factual_id)
		hit = result["_source"]
		return hit
	except:
		#print("Couldn't load factual merchant")
		return None

def safely_remove_file(filename):
	"""Safely removes a file"""
	print("Removing {0}".format(filename))
	try:
		os.remove(filename)
	except OSError:
		print("Unable to remove {0}".format(filename))
	print("File removed.")

def purge(my_dir, pattern):
	"""Cleans up processing location on System Exit"""
	for my_file in os.listdir(my_dir):
		if re.search(pattern, my_file):
			os.remove(os.path.join(my_dir, my_file))

def string_cleanse(original_string):
	"""Strips out characters that might confuse ElasticSearch."""
	bad_characters = [r"\[", r"\]", r"\{", r"\}", r'"', r"/", r"\\", r"\:", r"\(",\
	 r"\)", r"-", r"\+", r"<", r">", r"'", r"!", r"\*", r"\|\|", r"&&", r"~"]
	bad_character_regex = "|".join(bad_characters)
	cleanse_pattern = re.compile(bad_character_regex)
	with_spaces = re.sub(cleanse_pattern, " ", original_string)
	return ' '.join(with_spaces.split()).lower()

def build_boost_vectors(hyperparams):
	"""Turns field boosts into dictionary of numpy arrays"""

	boost_column_labels = hyperparams["boost_labels"]
	boost_row_vectors = hyperparams["boost_vectors"]
	boost_row_labels, boost_column_vectors = sorted(boost_row_vectors.keys()), {}

	for i in range(len(boost_column_labels)):

		my_list = []

		for field in boost_row_labels:
			my_list.append(boost_row_vectors[field][i])

		boost_column_vectors[boost_column_labels[i]] = np.array(my_list)

	return boost_row_labels, boost_column_vectors

def get_boosted_fields(hyperparams, vector_name):
	"""Returns a list of boosted fields built from a boost vector"""
	boost_row_labels, boost_column_vectors = build_boost_vectors(hyperparams)
	boost_vector = boost_column_vectors[vector_name]
	return [x + "^" + str(y)\
		for x, y in zip(boost_row_labels, boost_vector)\
		if y != 0.0]

def get_magic_query(params, transaction, boost=1.0):
	"""Build a magic query from pretrained boost vectors"""

	hyperparameters = load_hyperparameters(params)
	result_size = hyperparameters.get("es_result_size", "10")
	fields = params["output"]["results"]["fields"]
	good_description = transaction["GOOD_DESCRIPTION"]
	transaction = string_cleanse(transaction["DESCRIPTION_UNMASKED"]).rstrip()

	# Input transaction must not be empty
	if len(transaction) <= 2 and re.match('^[a-zA-Z0-9_]+$', transaction):
		return

	# Replace synonyms
	transaction = stopwords(transaction)
	transaction = synonyms(transaction)
	transaction = string_cleanse(transaction)

	# Construct Main Query
	magic_query = get_bool_query(size=result_size)
	magic_query["fields"] = fields
	magic_query["_source"] = "*"
	should_clauses = magic_query["query"]["bool"]["should"]
	field_boosts = get_boosted_fields(hyperparameters, "standard_fields")
	simple_query = get_qs_query(transaction, field_boosts, boost)
	should_clauses.append(simple_query)

	# Use Good Description in Query
	if good_description != ""\
		and hyperparameters.get("good_description", "") != "":
		good_description_boost = hyperparameters["good_description"]
		name_query = get_qs_query(string_cleanse(good_description),\
			['name'], good_description_boost)
		should_clauses.append(name_query)

	return magic_query

def get_bool_query(starting_from=0, size=0):
	"""Returns a bool style ElasticSearch query object"""
	return {
		"from" : starting_from,
		"size" : size,
		"query" : {
			"bool": {
				"minimum_number_should_match": 1,
				"should": []
			}
		}
	}

def get_qs_query(term, field_list=None, boost=1.0):
	"""Returns a "query_string" style ElasticSearch query object"""
	if field_list is None:
		field_list = []
	return {
		"query_string": {
			"query": term,
			"fields": field_list,
			"boost" : boost
		}
	}

def get_us_cities():
	"""Load an array of US cities"""
	with open("data/misc/US_Cities.txt") as city_file:
		cities = city_file.readlines()
	cities = [city.lower().rstrip('\n') for city in cities]
	return cities

def clean_bad_escapes(filepath):
	"""Clean a panel file of poorly escaped characters.
	Return false if required fields not present"""

	path, filename = os.path.split(filepath)
	filename = os.path.splitext(filename)[0]
	first_line = True
	required_fields = ["DESCRIPTION_UNMASKED", "UNIQUE_MEM_ID", "GOOD_DESCRIPTION"]

	# Clean File
	with gzip.open(filepath, "rb") as input_file:
		with open(path + "/" + filename, "wb") as output_file:
			for line in input_file:
				if first_line:
					for field in required_fields:
						if field not in str(line):
							safely_remove_file(filepath)
							safely_remove_file(path + "/" + filename)
							return False
					first_line = False
				line = clean_line(line)
				line = bytes(line + "\n", 'UTF-8')
				output_file.write(line)
	# Rename and Remove
	safely_remove_file(filepath)

	return filename

def clean_line(line):
	"""Strips out the part of a binary line that is not usable"""

	line = CLEAN_PATTERN.sub(" ", str(line)[2:-3])
	line = QUOTE_CLEAN.sub("", line)

	return line

def stopwords(transaction):
	"""Remove stopwords"""
	patterns = [
		r"^ach",
		r"\d{2}\/\d{2}",
		r"X{4}\d{4}"
		r"X{5}\d{4}",
		r"~{2}\d{5}~{2}\d{16}~{2}\d{5}~{2}\d~{4}\d{4}",
		r"checkcard \d{4}",
		r"\d{15}"
	]
	stop_words = [" pos ", r"^pos ", " ach ", "electronic", "debit",
		"purchase", " card ", " pin ", "recurring", " check ", "checkcard",
		"qps", "q35", "q03", " sq "]

	patterns = patterns + stop_words
	transaction = transaction.lower()
	regex = "|".join(patterns)
	cleanse_pattern = re.compile(regex)
	with_spaces = re.sub(cleanse_pattern, " ", transaction)
	return ' '.join(with_spaces.split()).upper()

def synonyms(transaction):
	"""Replaces transactions tokens with manually
	mapped factual representations. This method
	should be expanded to manage a file of synonyms"""
	rep = {
		"wal-mart" : " Walmart ",
		"wal mart" : " Walmart ",
		"samsclub" : " Sam's Club ",
		"usps" : " US Post Office ",
		"lowes" : " Lowe's ",
		"wholefds" : " Whole Foods ",
		"shell oil" : " Shell Gas ",
		"wm supercenter" : " Walmart ",
		"exxonmobil" : " exxonmobil exxon mobil ",
		"mcdonalds" : " mcdonald's ",
		"costco whse" : " costco ",
		"franciscoca" : " francisco ca ",
		"qt" : " Quicktrip ",
		"macy's east" : " Macy's ",
 	}

	transaction = transaction.lower()
	rep = dict((re.escape(k), v) for k, v in rep.items())
	pattern = re.compile("|".join(rep.keys()))
	text = pattern.sub(lambda m: rep[re.escape(m.group(0))], transaction)

	return text.upper()

def merge_split_files(params, split_list):
	"""Takes a split list and merges the files back together
	after processing is complete"""

	file_name = params["output"]["file"]["name"]
	base_path = params["output"]["file"]["processing_location"]
	full_path = base_path + file_name
	first_file = base_path + os.path.basename(split_list.pop(0))
	output = open(full_path, "a", encoding="utf-8")

	# Write first file with header
	with open(first_file, "r", encoding="utf-8") as head_file:
		for line in head_file:
			output.write(line)

	# Merge
	for split in split_list:
		base_file = os.path.basename(split)
		with open(base_path + base_file, 'r', encoding="utf-8") as chunk:
			next(chunk)
			for line in chunk:
				output.write(line)
		safely_remove_file(base_path + base_file)

	output.close()

	# GZIP
	unzipped = open(full_path, "rb")
	zipped = gzip.open(full_path + ".gz", "wb")
	zipped.writelines(unzipped)
	zipped.close()
	unzipped.close()

	# Cleanup
	safely_remove_file(first_file)
	safely_remove_file(full_path)

	return full_path + ".gz"

#Print a warning to not execute this file as a module
if __name__ == "__main__":
	print("This module is a library that contains useful functions;" \
		" it should not be run from the console.")
