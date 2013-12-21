#!/usr/bin/python
"""This tool creates an ElasticSearch index and bulk loads it with data."""

import json, sys, re
from various_tools import string_cleanse
from custom_exceptions import InvalidArguments, InvalidNumberOfLines\
, FileProblem
from query_templates import get_mapping_template

USAGE = """Usage:
	<input_file_name>
	<input_lines_to_scan>
	<elasticsearch_index>
	<elasticsearch_type>"""

SENTINEL = 200.0
NULL, FLOAT, DATE, INT, STRING = 0, 1, 2, 3, 4
DATA_TYPE_NAME, PATTERN = 0, 1
NAME, DATA_TYPE, INDEX = 0, 1, 2

def initialize():
	"""This function does basic validation for the command-line
	parameters.""" 
	if len(sys.argv) != 5:
		usage()
		raise InvalidArguments("Incorrect number of arguments")
	input_file_name, input_lines_to_scan, index, es_type = sys.argv[1:5]
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

	bulk_create_file = index + "." + es_type + ".bulk_create"
	try:
		bulk_create_file = open(bulk_create_file, "w")
	except:
		raise FileProblem(bulk_create_file + " cannot be created.")

	type_mapping_file = index + "." + es_type + ".mapping"
	try:
		type_mapping_file = open(type_mapping_file, "w")
	except:
		raise FileProblem(type_mapping_file + " cannot be created.")

	create_json_header = '{ "create" : { "_index" : "' + index \
	+ '", "_type": "' + es_type + '", "_id": "'
	return create_json_header, number_of_lines, input_file\
	, bulk_create_file, type_mapping_file, es_type

def revise_column_data_type(col_num, my_cell):
	"""This function tries to determine the best fit for the column,
	based upon observing values found in the input for the column."""
	my_data_type = COLUMN_META[col_num][DATA_TYPE]
	if my_data_type == STRING:
		return my_data_type
	if my_cell == "":
		return my_data_type
	if DATA_TYPES[my_data_type][PATTERN].match(my_cell):
		return my_data_type
	else:
		COLUMN_META[col_num][DATA_TYPE] += 1
		return revise_column_data_type(col_num, my_cell)
	
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

DATE_REGEX = "^[0-9]{4}(0[1-9]|1[012])(0[1-9]|[12][0-9]|3[01])$"

#DICTIONARIES
DATA_TYPES = { \
	NULL : ("null", re.compile("^$")) , \
	FLOAT : ("float", re.compile("[-+]?[0-9]*\.[0-9]+")) , \
	DATE : ("date", re.compile(DATE_REGEX)) , \
	INT : ("integer", re.compile("^[-+]?[0-9]+$")) , \
	STRING : ("string", re.compile(".+")) \
}

COLUMN_META = {}

LINE_COUNT = 0
LATITUDE, LONGITUDE = SENTINEL, SENTINEL

CREATE_JSON_HEADER, INPUT_LINES_TO_SCAN, INPUT_FILE, BULK_CREATE_FILE\
, TYPE_MAPPING_FILE, ES_TYPE = initialize()
CREATE_JSON_FOOTER = '" } }'

for line in INPUT_FILE:
	cells = line.split("\t")
	total_fields = len(cells)
	if LINE_COUNT == 0:
		COLUMN_META = scan_column_headers(cells)
	elif LINE_COUNT >= INPUT_LINES_TO_SCAN:
		break	
	else:
		create_api = CREATE_JSON_HEADER
		record = '{ '
		LATITUDE, LONGITUDE = SENTINEL, SENTINEL
		for column_number in range(len(cells)):
			cell = string_cleanse(str(cells[column_number]).strip())
			if column_number == 0:
				create_api += cell + CREATE_JSON_FOOTER
			revise_column_data_type(column_number, cell)

			#Exclude LATITUDE and LONGITUDE	until the end
			if COLUMN_META[column_number][NAME] == "LATITUDE":
				LATITUDE = cell	
			elif COLUMN_META[column_number][NAME] == "LONGITUDE":
				LONGITUDE = cell
			elif len(cell) == 0:
				continue
			else:
				record += '"'\
				+ COLUMN_META[column_number][NAME] + '": "'\
				+ cell + '", '

		#Add the geo-data, if there is any, otherwise just close
		#the document
		if len(str(LATITUDE)) > 0 and len(str(LONGITUDE)) > 0:
			record += '"pin" : { "location" : { "lat" : '\
			+ str(LATITUDE) + ', "lon" : ' + str(LONGITUDE)\
			+ ' } } }'
		else:
			record = record[0:-2] + "}"

		BULK_CREATE_FILE.write(create_api + "\n")	
		BULK_CREATE_FILE.write(record + "\n")
	LINE_COUNT += 1

MY_MAP = get_mapping_template(ES_TYPE, 3, 2, COLUMN_META, DATA_TYPES\
, total_fields)
TYPE_MAPPING_FILE.write(json.dumps(MY_MAP))
