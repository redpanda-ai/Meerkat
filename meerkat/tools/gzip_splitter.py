#!/bin/python3.3

import glob
import gzip
import logging
import os
import sys
import csv
from meerkat.various_tools import safely_remove_file

#Usage
# python3.3 -m meerkat.gzip_slicer <path_to_some_directory_of_gzipped_csv_files>

def split_csv(filehandler, **kwargs):
# pylint: disable=too-many-locals
	"""
	Adapted from Jordi Rivero:
	https://gist.github.com/jrivero
	Splits a CSV file into multiple pieces.

	A quick bastardization of the Python CSV library.

	Arguments:
	`row_limit`: The number of rows you want in each output file.
		10,000 by default.
	`output_name_template`: A %s-style template for the numbered output files.
	`output_path`: Where to stick the output files.
	`keep_headers`: Whether or not to print the headers in each output file.

	Example usage:
		>> from various_tools import split_csv;
		>> split_csv(open('/home/ben/input.csv', 'r'));
	"""
	delimiter = kwargs.get('delimiter', ',')
	row_limit = kwargs.get('row_limit', 10000)
	output_name_template = kwargs.get('output_name_template', 'output_%s.csv')
	output_path = kwargs.get('output_path', '.')
	keep_headers = kwargs.get('keep_headers', True)

	reader = csv.reader(filehandler, delimiter=delimiter)
	#Start at piece one
	current_piece = 1
	current_out_path = os.path.join(output_path,\
		output_name_template % current_piece)
	#Create a list of file pieces
	file_list = [current_out_path]
	current_out_writer = csv.writer(open(current_out_path, 'w',\
		encoding="utf-8"), delimiter=delimiter)
	current_limit = row_limit
	if keep_headers:
		headers = reader.__next__()
		current_out_writer.writerow(headers)
	#Split the file into pieces
	for i, row in enumerate(reader):
		if i + 1 > current_limit:
			current_piece += 1
			current_limit = row_limit * current_piece
			current_out_path = os.path.join(output_path,\
				output_name_template % current_piece)
			file_list.append(current_out_path)
			current_out_writer = csv.writer(open(current_out_path, 'w',\
				encoding="utf-8"), delimiter=delimiter)
			if keep_headers:
				current_out_writer.writerow(headers)
		current_out_writer.writerow(row)
	#Return complete list of chunks
	return file_list

def slice_me(working_directory, input_filename):
	os.chdir(working_directory)
	unzipped_filename = input_filename[:-3]
	#Gunzip the big file
	logging.critical("Gunzipping %s", input_filename)
	with gzip.open(input_filename, "rb") as zipped_input:
		with open(unzipped_filename, "wb") as unzipped_input:
			for line in zipped_input:
				unzipped_input.write(line)
	#Remove the big file
	safely_remove_file(working_directory + input_filename)

	split_list = None
	output_name_template = unzipped_filename + ".%s"
	#Split big unzipped file
	split_list = split_csv(open(unzipped_filename, 'r'), \
		delimiter='|', \
		row_limit=1000000,\
		output_name_template=output_name_template,\
		output_path=working_directory, \
		keep_headers=True)
	#Remove big unzipped file
	safely_remove_file(working_directory + unzipped_filename)
	for split in split_list:
		logging.critical("Gzipping %s", split)
		#gzip it
		with open(split, 'rb') as unzipped_input:
			with gzip.open(split + ".gz", "wb") as zipped_output:
				for line in unzipped_input:
					zipped_output.write(line)
		safely_remove_file(split)

os.chdir(sys.argv[1])

#Gather all txt.gz files
for current_file in sorted(glob.glob('*.txt.gz')):
	slice_me(sys.argv[1], current_file)
