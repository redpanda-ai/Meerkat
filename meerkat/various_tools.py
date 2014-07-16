#!/usr/local/bin/python3.3

"""This is where we keep functions that are useful 
enough to call from within multiple scripts.

Created on Dec 20, 2013
@author: J. Andrew Key
@author: Matthew Sevrens
"""

import csv
import re
import os
import gzip
import json

def load_dict_list(file_name, encoding='utf-8', delimiter="|"):
	"""Loads a dictionary of input from a file into a list."""
	input_file = open(file_name, encoding=encoding, errors='replace')
	dict_list = list(csv.DictReader(input_file, delimiter=delimiter,
		quoting=csv.QUOTE_NONE))
	input_file.close()
	return dict_list

def write_dict_list(dict_list, file_name, encoding="utf-8", delimiter="|"):
	""" Saves a lists of dicts with uniform keys to file """

	with open(file_name, 'w', encoding=encoding, errors='replace') as output_file:
		dict_w = csv.DictWriter(output_file, delimiter=delimiter, fieldnames=dict_list[0].keys(), extrasaction='ignore')
		dict_w.writeheader()
		dict_w.writerows(dict_list)

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
	es_connection = Elasticsearch(cluster_nodes, sniff_on_start=True,
	sniff_on_connection_fail=True, sniffer_timeout=15, sniff_timeout=15)

	return es_connection

def get_merchant_by_id(factual_id, es_connection, fields=["name", "region", "locality", "internal_store_number"]):
	"""Fetch the details for a single factual_id"""
	
	if factual_id == "NULL":
		return None

	try:
		result = es_connection.get(index="factual_index", doc_type='factual_type', id=factual_id)
		hit = result["_source"]
		return hit
	except:
		print("Couldn't get load factual merchant")
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
	original_string = original_string.replace("OR", "or")
	original_string = original_string.replace("AND", "and")
	bad_characters = [r"\[", r"\]", r"\{", r"\}", r'"', r"/", r"\\", r"\:",
		r"\(", r"\)", r"-", r"\+", r">", r"!", r"\*", r"\|\|", r"&&", r"~"]
	bad_character_regex = "|".join(bad_characters)
	cleanse_pattern = re.compile(bad_character_regex)
	with_spaces = re.sub(cleanse_pattern, " ", original_string)
	return ' '.join(with_spaces.split())

def synonyms(transaction):
	"""Replaces transactions tokens with manually
	mapped factual representations. This method
	should be expanded to manage a file of synonyms"""

	rep = {
		"wal-mart" : "Walmart",
		"samsclub" : "Sam's Club",
		"usps" : "US Post Office",
		"qps" : "",
		"q03" : "",
		"lowes" : "Lowe's",
		"wholefds" : "Whole Foods",
		"Shell Oil" : "Shell Gas",
		"wm supercenter" : "Walmart",
		"exxonmobil" : "exxonmobil exxon mobil",
		"mcdonalds" : "mcdonald's",
		"costco whse" : "costco",
		"franciscoca" : "francisco ca",
		"QT" : "Quicktrip",
		"Macy's East" : "Macy's"
 	}

	transaction = transaction.lower()
	rep = dict((re.escape(k), v) for k, v in rep.items())
	pattern = re.compile("|".join(rep.keys()))
	text = pattern.sub(lambda m: rep[re.escape(m.group(0))], transaction)

	return text

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
