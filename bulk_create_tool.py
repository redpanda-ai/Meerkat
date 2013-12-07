#!/usr/bin/python

import json, os, sys, re

def usage():
	print "Usage:\n\t<input_file_name> <input_lines_to_scan> <elasticsearch_index> <elasticsearch_type>"

class InvalidArguments(Exception):
	pass

class InvalidNumberOfLines(Exception):
	pass

class FileProblem(Exception):
	pass

SENTINEL = 200.0
NULL, FLOAT, DATE, INT, STRING = 0,1,2,3,4
DATA_TYPE_NAME, PATTERN = 0,1
NAME, DATA_TYPE, INDEX = 0,1,2

bad_characters = [ "\[", "\]", "'", "\{", "\}", '"']
x = "|".join(bad_characters)
cleanse_pattern = re.compile(x)

def initialize():
	if len(sys.argv) != 5:
		usage()
		raise InvalidArguments("Incorrect number of arguments")
	input_file_name,input_lines_to_scan,index,type = sys.argv[1:5]
	try:
		number_of_lines = int(input_lines_to_scan)
	except:
		usage()
		raise InvalidNumberOfLines("Number of lines must be an integer")
	try:
		INPUT_FILE = open(input_file_name)
	except:
		usage()
		raise FileProblem(input_file_name + " cannot be opened.")

	bulk_create_file = index + "." + type + ".bulk_create"
	try:
		BULK_CREATE_FILE = open(bulk_create_file,"w")
	except:
		raise FileProblem(bulk_create_file + " cannot be created.")

	type_mapping_file = index + "." + type + ".mapping"
	try:
		TYPE_MAPPING_FILE = open(type_mapping_file,"w")
	except:
		raise FileProblem(type_mapping_file + " cannot be created.")

	create_json_header = '{ "create" : { "_index" : "' + index + '", "_type": "' + type + '", "_id": "'
	return create_json_header, number_of_lines, INPUT_FILE, BULK_CREATE_FILE, TYPE_MAPPING_FILE

def revise_column_data_type(i,cell):
	my_data_type = column_meta[i][DATA_TYPE]
	if my_data_type == STRING:
		return my_data_type
	if cell == "":
		return my_data_type
	if data_types[my_data_type][PATTERN].match(cell):
		return my_data_type
	else:
		column_meta[i][DATA_TYPE] += 1
		return revise_column_data_type(i,cell)
	
def scan_column_headers(cells):
	column_meta = {}
	for i in range(len(cells)):
		#NAME, DATA_TYPE, INDEX
		column_meta[i] = [cells[i].strip(), NULL, "analyzed" ]
	return column_meta

def string_cleanse(s):
	return re.sub(cleanse_pattern,"",s)	


#DICTIONARIES
data_types = { \
	NULL : ("null",re.compile("^$")) , \
	FLOAT : ("float", re.compile("[-+]?[0-9]*\.[0-9]+")) , \
	DATE : ("date", re.compile("^[0-9]{4}(0[1-9]|1[012])(0[1-9]|[12][0-9]|3[01])$")) , \
	INT : ("integer",re.compile("^[-+]?[0-9]+$")) , \
	STRING : ("string", re.compile(".+")) \
}

column_meta = {}

line_count = 0
global latitude
global longitude
latitude, longitude = SENTINEL, SENTINEL

create_json_header, input_lines_to_scan, INPUT_FILE, BULK_CREATE_FILE, TYPE_MAPPING_FILE = initialize()
create_json_footer = '" } }'

for line in INPUT_FILE:
	cells = line.split("\t")
	total_fields = len(cells)
	if line_count == 0:
		column_meta = scan_column_headers(cells)
	elif line_count >= input_lines_to_scan:
		break	
	else:
		create_api = create_json_header
		record = '{ '
		latitude, longitude = SENTINEL, SENTINEL
		for i in range(len(cells)):
			cell = string_cleanse(str(cells[i]).strip())
			if i == 0:
				create_api += cell + create_json_footer
			revise_column_data_type(i,cell)


			#Exclude latitude and longitude	until the end
			if column_meta[i][NAME] == "LATITUDE":
				latitude = cell	
			elif column_meta[i][NAME] == "LONGITUDE":
				longitude = cell
			elif len(cell) == 0:
				continue
			else:
				record += '"' + column_meta[i][NAME] + '": "' + cell + '", '

		#Add the geo-data, if there is any, otherwise just close the document
		if len(str(latitude)) > 0 and len(str(longitude)) > 0:
			record += '"pin" : { "location" : { "lat" : ' + str(latitude) + ', "lon" : ' + str(longitude) + ' } } }'
		else:
			record = record[0:-2] + "}"

		BULK_CREATE_FILE.write(create_api + "\n")	
		BULK_CREATE_FILE.write(record + "\n")
	line_count += 1

mappings = """
{ "settings" : { "number_of_shards" : 3, "number_of_replicas" : 2 },
"mappings" : { "merchant" : { "_source" : { "enabled" : true }, "properties" : {
"""

for i in range(total_fields):
	my_data_type = column_meta[i][DATA_TYPE]
	column_name = column_meta[i][NAME]
	column_type = data_types[my_data_type][DATA_TYPE_NAME]
	column_index = column_meta[i][INDEX]
	format_string = '' 
	if column_type == "date" :
		format_string = ', "format" : "YYYYmmdd"' 
	if column_type in [ "null" ]:
		pass
	elif column_name in ["LATITUDE","LONGITUDE"]:
		pass
	else:
		mappings += '"' + column_name + '" : {"type" : "' + column_type + '", "index" : "' + column_index + '"' + format_string + '},'

mappings += '"pin" : { "properties" : { "location" : { "type" : "geo_point" } } }' + ' } } } }'
TYPE_MAPPING_FILE.write(mappings)
