#!/usr/bin/python
"""This tool creates an ElasticSearch index and bulk loads it with data."""

import json, sys, re
from various_tools import string_cleanse
from custom_exceptions import InvalidArguments, InvalidNumberOfLines\
, FileProblem
from query_templates import get_mapping_template, get_create_object

USAGE = """Usage:
	<input_file_name>
	<input_lines_to_scan>
	<elasticsearch_index>
	<elasticsearch_type>"""

NAME, DATA_TYPE, INDEX = 0, 1, 2

def initialize():
	"""This function does basic validation for the command-line
	parameters.""" 
	if len(sys.argv) != 5:
		usage()
		raise InvalidArguments("Incorrect number of arguments")
	input_file_name, input_lines_to_scan, es_index, es_type = sys.argv[1:5]
	try:
		number_of_lines = int(input_lines_to_scan)
	except:
		usage()
		raise InvalidNumberOfLines("Number of lines must be an integer")
	try:
		input_file = open(input_file_name)
	except:
		usage()
		raise FileProblem(input_file_name + " cannot be opened.")

	bulk_create_file = es_index + "." + es_type + ".bulk_create"
	try:
		bulk_create_file = open(bulk_create_file, "w")
	except:
		raise FileProblem(bulk_create_file + " cannot be created.")

	type_mapping_file = es_index + "." + es_type + ".mapping"
	try:
		type_mapping_file = open(type_mapping_file, "w")
	except:
		raise FileProblem(type_mapping_file + " cannot be created.")

	return number_of_lines, input_file , bulk_create_file\
	, type_mapping_file, es_index, es_type

def revise_column_data_type(col_num, my_cell, column_meta):
	"""This function tries to determine the best fit for the column,
	based upon observing values found in the input for the column."""
	pattern = 1
	my_data_type = column_meta[col_num][DATA_TYPE]
	if my_data_type == STRING:
		return my_data_type
	if my_cell == "":
		return my_data_type
	if DATA_TYPES[my_data_type][pattern].match(my_cell):
		return my_data_type
	else:
		column_meta[col_num][DATA_TYPE] += 1
		return revise_column_data_type(col_num, my_cell, column_meta)
	
def scan_column_headers(my_cells):
	"""This function scans the first line of the data input file in 
	order to provide column names for our index."""
	column_meta = {}
	for my_col_number in range(len(my_cells)):
		#NAME, DATA_TYPE, INDEX
		column_meta[my_col_number] = [my_cells[my_col_number].strip()
		, NULL, "analyzed" ]
	return column_meta

def usage():
	"""Shows which command line arguments should be passed to the
	program."""
	print USAGE

def process_row(cells, column_meta, es_index, es_type):
	"""Scans a row from the input and returns:
	1.  JSON for the 'bulk create'
	2.  A record object that contains most fields needed"""
	record_obj = {}
	for column_number in range(len(cells)):
		cell = string_cleanse(str(cells[column_number]).strip())
		if column_number == 0:
			create_obj = get_create_object(es_index, es_type, cell)
			create_json = json.dumps(create_obj)
		revise_column_data_type(column_number, cell, column_meta)

		#Exclude latitude and longitude	until the end
		if column_meta[column_number][NAME] in \
		("LATITUDE", "LONGITUDE"):
			continue
		elif len(cell) == 0:
			continue
		else:
			record_obj[column_meta[column_number][NAME]] = cell
	return record_obj, create_json

def process_input_rows(input_file, es_index, es_type):
	"""Reads each line in the input file, creating bulk insert records
	for each in ElasticSearch."""
	line_count = 0
	sentinel = 200.0
	latitude, longitude = sentinel, sentinel
	column_meta = {}
	for line in input_file:
		cells = line.split("\t")
		column_meta["total_fields"] = len(cells)
		if line_count == 0:
			column_meta = scan_column_headers(cells)
		elif line_count >= INPUT_LINES_TO_SCAN:
			break	
		else:
			#create_api = create_json_header
			latitude, longitude = sentinel, sentinel
			record_obj, create_json = \
			process_row(cells, column_meta , es_index, es_type)
			#Add the geo-data, if there is any
			if len(str(latitude)) > 0 and len(str(longitude)) > 0:
				record_obj["pin"] = {}
				record_obj["pin"]["location"] = {}
				location_obj = record_obj["pin"]["location"] 
				location_obj["lat"] = str(latitude)
				location_obj["lon"] = str(latitude)
			#This would be where we add composite fields
			record_json = json.dumps(record_obj)	
			BULK_CREATE_FILE.write(create_json + "\n")	
			BULK_CREATE_FILE.write(record_json + "\n")
		line_count += 1
	return column_meta

#DICTIONARIES
NULL, FLOAT, DATE, INT, STRING = 0, 1, 2, 3, 4
DATE_REGEX = "^[0-9]{4}(0[1-9]|1[012])(0[1-9]|[12][0-9]|3[01])$"
DATA_TYPES = { \
	NULL : ("null", re.compile("^$")) , \
	FLOAT : ("float", re.compile("[-+]?[0-9]*\.[0-9]+")) , \
	DATE : ("date", re.compile(DATE_REGEX)) , \
	INT : ("integer", re.compile("^[-+]?[0-9]+$")) , \
	STRING : ("string", re.compile(".+")) \
}

INPUT_LINES_TO_SCAN, INPUT_FILE, BULK_CREATE_FILE, TYPE_MAPPING_FILE\
, ES_INDEX, ES_TYPE = initialize()
COLUMN_META = process_input_rows(INPUT_FILE, ES_INDEX, ES_TYPE)
MY_MAP = get_mapping_template(ES_TYPE, 3, 2, COLUMN_META, DATA_TYPES)
TYPE_MAPPING_FILE.write(json.dumps(MY_MAP))
