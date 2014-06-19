#!/bin/python3.3

import glob
import gzip
import logging
import os
import sys
from random import shuffle
from .various_tools import safely_remove_file, split_csv


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
	split_list = split_csv(open(unzipped_filename, 'r'), delimiter='|', row_limit=1000000,\
		output_name_template=output_name_template,\
		output_path=working_directory, keep_headers=True)
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

working_directory = sys.argv[1]
os.chdir(working_directory)

#Gather all txt.gz files
input_files = sorted(glob.glob('*.txt.gz'))
for current_file in input_files:
	slice_me(working_directory, current_file)
