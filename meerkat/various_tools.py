#!/usr/local/bin/python3.3

"""This is where we keep functions that are useful 
enough to call from within multiple scripts.

Created on Dec 20, 2013
@author: J. Andrew Key
@author: Matthew Sevrens
"""

import csv
import sys
import re
import os
import gzip
import json

import numpy as np

CLEAN_PATTERN = re.compile(r"\\+\|")

def load_dict_list(file_name, encoding='utf-8', delimiter="|"):
	"""Loads a dictionary of input from a file into a list."""
	input_file = open(file_name, encoding=encoding, errors='replace')
	dict_list = list(csv.DictReader(input_file, delimiter=delimiter,
		quoting=csv.QUOTE_NONE))
	input_file.close()
	return dict_list

def load_dict_ordered(file_name, encoding='utf-8', delimiter="|"):
	"""Loads a dictionary of input, anf returns a list and ordered
	fieldnames"""

	input_file = open(file_name, encoding=encoding, errors='replace')
	reader = csv.DictReader(input_file, delimiter=delimiter, quoting=csv.QUOTE_NONE)
	dict_list = list(reader)
	input_file.close()
	return dict_list, reader.fieldnames

def write_dict_list(dict_list, file_name, encoding="utf-8", delimiter="|", column_order=""):
	""" Saves a lists of dicts with uniform keys to file """

	if column_order == "":
		column_order = dict_list[0].keys()

	with open(file_name, 'w', encoding=encoding, errors='replace') as output_file:
		dict_w = csv.DictWriter(output_file, delimiter=delimiter, fieldnames=column_order, extrasaction='ignore')
		dict_w.writeheader()
		dict_w.writerows(dict_list)

def get_panel_header(container):
	"""Return an ordered consistent header for panels"""

	header = [\
		"UNIQUE_MEM_ID", "UNIQUE___BLANK_ACCOUNT_ID", "UNIQUE___BLANK_TRANSACTION_ID",\
		"MEM_ID", "__BLANK_ACCOUNT_ID", "__BLANK_TRANSACTION_ID", "COBRAND_ID",\
		"SUM_INFO_ID", "AMOUNT", "CURRENCY_ID", "DESCRIPTION", "TRANSACTION_DATE",\
		"POST_DATE", "TRANSACTION_BASE_TYPE", "TRANSACTION_CATEGORY_ID",\
		"TRANSACTION_CATEGORY_NAME", "MERCHANT_NAME", "STORE_ID", "FACTUAL_CATEGORY",\
		"STREET", "CITY", "STATE", "ZIP_CODE", "WEBSITE", "PHONE_NUMBER", "FAX_NUMBER",\
		"CHAIN_NAME", "LATITUDE", "LONGITUDE", "NEIGHBOURHOOD", "TRANSACTION_ORIGIN",\
		"CONFIDENCE_SCORE", "FACTUAL_ID", "FILE_CREATED_DATE", "DESCRIPTION_UNMASKED",\
		"GOOD_DESCRIPTION"
	]

	container = container.upper()
	header = [x.replace("__BLANK", container) for x in header]
	return header

def get_yodlee_factual_map():
	"""Return a map of factual attribute names to
	yodlee attribute names"""

	return {
		"factual_id" : "FACTUAL_ID",
		"address" : "STREET",
		"tel" : "PHONE_NUMBER",
		"latitude" : "LATITUDE",
		"longitude" : "LONGITUDE",
		"postcode" : "ZIP_CODE",
		"region" : "STATE",
		"website" : "WEBSITE",
		"locality" : "CITY",
		"z_score_delta" : "CONFIDENCE_SCORE",
		"category_labels" : "FACTUAL_CATEGORY",
		"chain_name" : "CHAIN_NAME",
		"neighborhood" : "NEIGHBOURHOOD",
		"internal_store_number" : "STORE_ID",
		"name" : "MERCHANT_NAME"
	}

def get_column_map(container):
	"""Fix old or erroneous column names"""

	return {
		"UNIQUE_ACCOUNT_ID" : "UNIQUE_" + container.upper() + "_ACCOUNT_ID",
		"UNIQUE_TRANSACTION_ID" : "UNIQUE_" + container.upper() + "_TRANSACTION_ID",
		"TYPE" : "TRANSACTION_BASE_TYPE",
		"MERCHANT_NAME" : "GOOD_DESCRIPTION"
	}

def get_new_columns():
	"""Return a list of new columns to add to a panel"""

	return [
		'STORE_ID', 
		'FACTUAL_CATEGORY', 
		'STREET', 
		'CITY', 
		'STATE', 
		'ZIP_CODE', 
		'WEBSITE', 
		'PHONE_NUMBER', 
		'FAX_NUMBER', 
		'CHAIN_NAME', 
		'LATITUDE', 
		'LONGITUDE',
		'MERCHANT_NAME',
		'NEIGHBOURHOOD', 
		'TRANSACTION_ORIGIN', 
		'CONFIDENCE_SCORE', 
		'FACTUAL_ID'
	]

def to_stdout(string, errors="replace"):
	"""Converts a string to stdout compatible encoding"""

	encoded = string.encode(sys.stdout.encoding, errors)
	decoded = encoded.decode(sys.stdout.encoding)
	return decoded

def safe_print(*objs, errors="replace"):
	"""Print without unicode errors"""
	
	print(*(to_stdout(str(o), errors) for o in objs))

def progress(i, list, message=""):
	"""Display progress percent in a loop"""

	progress = (i / len(list)) * 100
	progress = str(round(progress, 1)) + "% " + message 
	sys.stdout.write('\r')
	sys.stdout.write(progress)
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

	input_file = open(filename, encoding='utf-8')
	params = json.loads(input_file.read())
	input_file.close()

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

def get_es_connection(params):
	"""Fetch a connection to the factual index"""

	from elasticsearch import Elasticsearch

	cluster_nodes = params["elasticsearch"]["cluster_nodes"]
	index = params["elasticsearch"]["index"]
	es_connection = Elasticsearch(cluster_nodes, index=index, sniff_on_start=True,
	sniff_on_connection_fail=True, sniffer_timeout=15, sniff_timeout=15)

	return es_connection

def get_merchant_by_id(params, factual_id, es_connection, index=""):
	"""Fetch the details for a single factual_id"""

	if index == "":
		index = params.get("elasticsearch", {}).get("index", "")
	
	if factual_id == "NULL":
		return None

	try:
		result = es_connection.get(index=index, doc_type='factual_type', id=factual_id)
		hit = result["_source"]
		return hit
	except:
		#print("Couldn't load factual merchant")
		return None

def numeric_cleanse(original_string):
	"""Strips out characters that might confuse ElasticSearch."""
	bad_characters = [r"\[", r"\]", r"'", r"\{", r"\}", r'"', r"/", r"-"]
	bad_character_regex = "|".join(bad_characters)
	cleanse_pattern = re.compile(bad_character_regex)
	return re.sub(cleanse_pattern, "", original_string)

def safely_remove_file(filename):
	"""Safely removes a file"""
	print("Removing {0}".format(filename))
	try:
		os.remove(filename)
	except OSError:
		print("Unable to remove {0}".format(filename))
	print("File removed.")

def purge(dir, pattern):
	"""Cleans up processing location on System Exit"""
	for f in os.listdir(dir):
		if re.search(pattern, f):
			os.remove(os.path.join(dir, f))

def string_cleanse(original_string):
	"""Strips out characters that might confuse ElasticSearch."""
	bad_characters = [r"\[", r"\]", r"\{", r"\}", r'"', r"/", r"\\", r"\:",
		r"\(", r"\)", r"-", r"\+", r">", r"!", r"\*", r"\|\|", r"&&", r"~"]
	bad_character_regex = "|".join(bad_characters)
	cleanse_pattern = re.compile(bad_character_regex)
	with_spaces = re.sub(cleanse_pattern, " ", original_string)
	return ' '.join(with_spaces.split()).lower()

def build_boost_vectors(params):
	"""Turns field boosts into dictionary of numpy arrays"""

	boost_column_labels = params["elasticsearch"]["boost_labels"]
	boost_row_vectors = params["elasticsearch"]["boost_vectors"]
	boost_row_labels, boost_column_vectors = sorted(boost_row_vectors.keys()), {}

	for i in range(len(boost_column_labels)):

		my_list = []

		for field in boost_row_labels:
			my_list.append(boost_row_vectors[field][i])

		boost_column_vectors[boost_column_labels[i]] = np.array(my_list)

	return boost_row_labels, boost_column_vectors

def get_boosted_fields(params, vector_name):
	"""Returns a list of boosted fields built from a boost vector"""

	boost_row_labels, boost_column_vectors = build_boost_vectors(params)
	boost_vector = boost_column_vectors[vector_name]
	return [x + "^" + str(y) for x, y in zip(boost_row_labels, boost_vector) if y != 0.0]

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
	field_boosts = get_boosted_fields(params, "standard_fields")
	simple_query = get_qs_query(transaction, field_boosts, boost)
	should_clauses.append(simple_query)

	# Use Good Description in Query
	if good_description != "" and hyperparameters.get("good_description", "") != "":
		good_description_boost = hyperparameters["good_description"]
		name_query = get_qs_query(string_cleanse(good_description), ['name'], good_description_boost)
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

def get_qs_query(term, field_list=[], boost=1.0):
	"""Returns a "query_string" style ElasticSearch query object"""

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
	with gzip.open(filepath, "rb") as f:
		with open(path + "/" + filename, "wb") as g:
			for l in f:
				if first_line:
					for field in required_fields:
						if field not in str(l):
							safely_remove_file(filepath)
							safely_remove_file(path + "/" + filename)
							return False
				line = clean_line(l)
				line = bytes(line + "\n", 'UTF-8')
				g.write(line)

	# Rename and Remove
	safely_remove_file(filepath)

	return filename

def clean_line(line):
	"""Strips out the part of a binary line that is not usable"""
	return CLEAN_PATTERN.sub(" ", str(line)[2:-3])

def stopwords(transaction):
	"""Remove stopwords"""

	patterns = [
		r"^ach", 
		r"\d{2}\/\d{2}", 
		r"X{4}\d{4}",
		r"X{5}\d{4}", 
		r"~{2}\d{5}~{2}\d{16}~{2}\d{5}~{2}\d~{4}\d{4}", 
		r"checkcard \d{4}", 
		r"\d{15}"
	]

	stop_words = [
		" pos ",
		r"^pos ",  
		" ach ", 
		"electronic", 
		"debit", 
		"purchase", 
		" card ", 
		" pin ", 
		"recurring", 
		" check ",
		"checkcard",
		"qps",
		"q35",
		"q03",
		" sq "
	]

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

def split_csv(filehandler, delimiter=',', row_limit=10000, 
	output_name_template='output_%s.csv', output_path='.', keep_headers=True):
	"""
	Adapted from Jordi Rivero:
	https://gist.github.com/jrivero
	Splits a CSV file into multiple pieces.
	
	A quick bastardization of the Python CSV library.

	Arguments:
		`row_limit`: The number of rows you want in each output file. 10,000 by default.
		`output_name_template`: A %s-style template for the numbered output files.
		`output_path`: Where to stick the output files.
		`keep_headers`: Whether or not to print the headers in each output file.

	Example usage:
		>> from various_tools import split_csv;
		>> split_csv(open('/home/ben/input.csv', 'r'));
	
	"""
	reader = csv.reader(filehandler, delimiter=delimiter)
	#Start at piece one
	current_piece = 1
	current_out_path = os.path.join(
		 output_path,
		 output_name_template  % current_piece
	)
	#Create a list of file pieces
	file_list = [current_out_path]
	current_out_writer = csv.writer(open(current_out_path, 'w', encoding="utf-8"), delimiter=delimiter)
	current_limit = row_limit
	if keep_headers:
		headers = reader.__next__()
		current_out_writer.writerow(headers)
	#Split the file into pieces
	for i, row in enumerate(reader):
		if i + 1 > current_limit:
			current_piece += 1
			current_limit = row_limit * current_piece
			current_out_path = os.path.join( output_path, output_name_template  % current_piece)
			file_list.append(current_out_path)
			current_out_writer = csv.writer(open(current_out_path, 'w', encoding="utf-8"), delimiter=delimiter)
			if keep_headers:
				current_out_writer.writerow(headers)
		current_out_writer.writerow(row)
	#Return complete list of chunks
	return file_list

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

if __name__ == "__main__":
	"""Print a warning to not execute this file as a module"""
	print("This module is a library that contains useful functions; it should not be run from the console.")
