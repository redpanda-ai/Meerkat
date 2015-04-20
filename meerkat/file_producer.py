#!/usr/local/bin/python3.3

"""This module loads and processes transactions in preparation for
analysis and search though our structured ElasticSearch merchant index.

Created on Dec 9, 2013
@author: J. Andrew Key
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3.3 -m meerkat.file_producer <location_pair_name> <file_name>
# python3.3 -m meerkat.file_producer p1 20150205.007.012_P1_BANK.txt.gz

#####################################################

import boto
import csv
import gzip
import io
import json
import logging
import os
import pandas as pd
import queue
import re
import sys

from boto.s3.connection import Key, Location
from jsonschema import validate
from jsonschema.exceptions import ValidationError, SchemaError

from meerkat.custom_exceptions import InvalidArguments
from meerkat.file_consumer import FileConsumer
from meerkat.classification.load import select_model
from meerkat.various_tools import (safely_remove_file)
from meerkat.various_tools import (get_us_cities, post_SNS)

#CONSTANTS
USED_IN_HEADER, ORIGIN, NAME_IN_MEERKAT, NAME_IN_ORIGIN = 0, 1, 2, 3

#Allowed pylint exception
# pylint: disable=bad-continuation

def get_field_mappings(params):
	"""Returns a list of field_mappings."""
	return [[x[NAME_IN_ORIGIN], x[NAME_IN_MEERKAT]]
		for x in get_unified_header(params)
		if (x[ORIGIN] == "search") and (x[NAME_IN_MEERKAT] != x[NAME_IN_ORIGIN])]

def get_meerkat_fields(params):
	"""Return a list of meerkat fields to add to the panel output."""
	return [x[NAME_IN_MEERKAT]
		for x in get_unified_header(params)
		if (x[USED_IN_HEADER] == True) and (x[ORIGIN] == "search")]

def get_column_map(params):
	"""Fix old or erroneous column names"""
	container = params["container"].upper()
	column_mapping_list = [
		(x[NAME_IN_ORIGIN], x[NAME_IN_MEERKAT].replace("__BLANK", container))
		for x in get_unified_header(params)
		if (x[ORIGIN] == "input") and (x[NAME_IN_MEERKAT] != x[NAME_IN_ORIGIN])]
	column_map = {}
	for name_in_origin, name_in_meerkat in column_mapping_list:
		column_map[name_in_origin] = name_in_meerkat
	return column_map

def get_panel_header(params):
	"""Return an ordered consistent header for panels"""
	return [
		x[NAME_IN_MEERKAT].replace("__BLANK", params["container"].upper())
		for x in get_unified_header(params)]

def get_unified_header(params):
	"""Return the unified_header object, minus the first row."""
	return params["my_producer_options"]["unified_header"][1:]

def clean_dataframe(params, dataframe):
	"""Fix issues with current dataframe, like remapping, etc."""
	column_remap = get_column_map(params)
	header = get_panel_header(params)
	meerkat_fields = get_meerkat_fields(params)
	# Rename and add columns
	dataframe = dataframe.rename(columns=column_remap)
	for meerkat_field in meerkat_fields:
		dataframe[meerkat_field] = ""
	# Reorder header
	dataframe = dataframe[header]
	return dataframe

def get_s3_connection():
	"""Returns a connection to S3"""
	try:
		conn = boto.connect_s3()
	except boto.s3.connection.HostRequiredError:
		print("Error connecting to S3, check your credentials")
		sys.exit()
	return conn

def convert_dataframe_to_queue(params, dataframe):
	"""Converts a dataframe to a queue for processing"""
	container = params["container"]
	#Pull the correct model
	classifier = select_model(container)
	desc_queue = queue.Queue()
	name_map = {"GOOD_DESCRIPTION" : "MERCHANT_NAME",\
		"MERCHANT_NAME" : "GOOD_DESCRIPTION"}
	# Classify transactions
	classes = ["Non-Physical", "Physical", "ATM"]
	get_classes = lambda x: classes[int(classifier(x["DESCRIPTION_UNMASKED"]))]
	dataframe['TRANSACTION_ORIGIN'] = dataframe.apply(get_classes, axis=1)
	my_groupby = dataframe.groupby('TRANSACTION_ORIGIN')
	groups = list(my_groupby.groups)
	# Group into separate dataframes
	physical = pd.DataFrame()
	if "Physical" in groups:
		physical = my_groupby.get_group("Physical")
	atm = pd.DataFrame()
	if "ATM" in groups:
		atm = my_groupby.get_group("ATM")
	non_physical = pd.DataFrame()
	if "Non-Physical" in groups:
		non_physical = my_groupby.get_group("Non-Physical").rename(columns=name_map)
	#.Return if there are no physical transactions
	if physical.empty:
		return desc_queue, non_physical
	# Concatenate ATM onto physical
	physical = pd.concat([physical, atm])
	# Group by user
	user_groupby = physical.groupby('UNIQUE_MEM_ID')
	for user in user_groupby:
		user_trans = []
		for _, row in user[1].iterrows():
			user_trans.append(row.to_dict())
		desc_queue.put(user_trans)
	return desc_queue, non_physical

def get_container(filename):
	"""Parses the filename to determine if the file is either
		for 'bank' or 'card' transactions."""
	container = None
	if "bank" in filename.lower():
		container = "bank"
	elif "card" in filename.lower():
		container = "card"
	if container:
		logging.warning("{0} discovered as container.".format(container))
		return container
	logging.error("Neither 'bank' nor 'card' present in file name, aborting.")
	#TODO Add a proper exception
	sys.exit()

def get_panel_header_old(params):
	"""Return an ordered consistent header for panels"""
	header = params["my_producer_options"]["header_template"]
	container = params["container"].upper()
	header = [x.replace("__BLANK", container) for x in header]
	return header

def gunzip_and_validate_file(filepath):
	"""Decompress a file, check for required fields on the first line."""
	logging.warning("Gunzipping and validating {0}".format(filepath))
	path, filename = os.path.split(filepath)
	filename = os.path.splitext(filename)[0]
	first_line = True
	required_fields = ["DESCRIPTION_UNMASKED", "UNIQUE_MEM_ID", "GOOD_DESCRIPTION"]
	# Clean File
	with gzip.open(filepath, "rt") as input_file:
		with open(path + "/" + filename, "wt") as output_file:
			for line in input_file:
				if first_line:
					for field in required_fields:
						if field not in str(line):
							safely_remove_file(filepath)
							safely_remove_file(path + "/" + filename)
							logging.error("Required fields not found in header, aborting")
							#Write to the error file first
							sys.exit()
					first_line = False
				output_file.write(line)
	# Remove original file
	safely_remove_file(filepath)
	logging.warning("{0} unzipped; header contains mandatory fields."
		.format(filename))
	return filename

def set_custom_producer_options(params):
	"""Applies overrides specific to a particular panel to the default."""
	my_producer_options = params["producer_options"]
	params["my_producer_options"] = my_producer_options["producer_default"]
	if sys.argv[1] in my_producer_options:
		overrides = my_producer_options[sys.argv[1]]
		for key in overrides:
			params["my_producer_options"][key] = overrides[key]
	del params["producer_options"]

def initialize():
	"""Validates the command line arguments."""
	input_file, params = None, None
	if len(sys.argv) != 3:
		usage()
		raise InvalidArguments(msg="Incorrect number of arguments", expr=None)
	try:
		input_file = open("config/daemon/file.json", encoding='utf-8')
		params = json.loads(input_file.read())
		input_file.close()
	except IOError:
		logging.error("Configuration file not found, aborting.")
		sys.exit()
	found = False
	for item in params["location_pairs"]:
		if item["name"] == sys.argv[1]:
			item["err_location"] = params["err_location"]
			params["location_pair"] = item
			del params["location_pairs"]
			found = True
			break
	if not found:
		raise InvalidArguments(
			msg="Invalid 'location_pair' argument, aborting.", expr=None)
	logging.warning("location_pair found in configuration file.")
	params["src_file"] = sys.argv[2]
	set_custom_producer_options(params)
	#Adding other fields for compatiblity with various tools library
	my_options = params["my_producer_options"]
	params["output"] = my_options["output"]
	params["output"]["results"] = {}
	params["output"]["results"]["fields"] = []
	params["my_producer_options"]["field_mappings"] = get_field_mappings(params)
	for field, _ in my_options["field_mappings"]:
		params["output"]["results"]["fields"].append(field)
	params["mode"] = "production"
	return params

def pull_src_file_from_s3(params):
	"""Pulls a source file from S3 and delivers it to the local host."""
	#Grab relevant configuraiton parameters
	src_s3_location = params["location_pair"]["src_location"]
	location_pattern = re.compile("^([^/]+)/(.*)$")
	matches = location_pattern.search(src_s3_location)
	bucket_name = matches.group(1)
	directory = matches.group(2)
	src_file = params["src_file"]
	logging.warning("S3 Bucket: {0}, S3 directory: {1}, Src file: {2}".
		format(bucket_name, directory, src_file))

	#Pull the src file from S3
	conn = get_s3_connection()
	bucket = conn.get_bucket(bucket_name, Location.USWest2)
	listing = bucket.list(prefix=directory+src_file)
	s3_key = None
	for item in listing:
		s3_key = item
	params["local_src_path"] = \
		params["my_producer_options"]["local_files"]["src_path"]
	local_src_file = params["local_src_path"] + src_file
	s3_key.get_contents_to_filename(local_src_file)
	logging.warning("Src file pulled from S3")
	return local_src_file

def push_dst_file_to_s3(params):
	#TODO: Refactor me
	"""Moves a file to S3"""
	dst_s3_location = params["location_pair"]["dst_location"]
	location_pattern = re.compile("^([^/]+)/(.*)$")
	matches = location_pattern.search(dst_s3_location)
	bucket_name = matches.group(1)
	directory = matches.group(2)
	dst_file = params["src_file"]
	#Push the dst file to S3
	conn = get_s3_connection()
	bucket = conn.get_bucket(bucket_name, Location.USWest2)
	key = Key(bucket)
	key.key = directory + dst_file
	bytes_written = key.set_contents_from_filename(
		params["local_gzipped_dst_file"], encrypt_key=True, replace=True)
	logging.warning("{0} pushed to S3, {1} bytes written.".format(
		dst_file, bytes_written))

def push_file_to_s3(params, type):
	"""Moves a file to S3"""
	type_location = type + "_location"
	gzipped_type_file = "local_gzipped_" + type + "_file"
	s3_type_location = params["location_pair"][type_location]
	location_pattern = re.compile("^([^/]+)/(.*)$")
	matches = location_pattern.search(s3_type_location)
	bucket_name = matches.group(1)
	directory = matches.group(2)
	type_file = params["src_file"]
	#Push the dst file to S3
	conn = get_s3_connection()
	bucket = conn.get_bucket(bucket_name, Location.USWest2)
	key = Key(bucket)
	key.key = directory + type_file
	bytes_written = key.set_contents_from_filename(
		params[gzipped_type_file], encrypt_key=True, replace=True)
	logging.warning("{0} pushed to S3, {1} bytes written.".format(
		type_file, bytes_written))

def process_single_input_file(params):
	"""Runs Meerkat in production mode on a single file"""
	#Pull the file from S3
	local_gzipped_src_file = pull_src_file_from_s3(params)
	# Gunzip the file, confirm that the required fields are included
	src_file_name = gunzip_and_validate_file(local_gzipped_src_file)
	# Discover the container type
	params["container"] = get_container(src_file_name)
	# Set destination filename for results
	dst_file_name = src_file_name
	local_src_file = params["local_src_path"] + src_file_name
	# Get a pandas dataframe ready
	reader = pd.read_csv(local_src_file, na_filter=False, chunksize=5000,
		quoting=csv.QUOTE_NONE, encoding="utf-8", sep='|',
		error_bad_lines=False)
	logging.warning("Dataframe reader loaded.")
	# Process a single file with Meerkat
	local_dst_file = run_panel(params, reader, dst_file_name)
	# Write panel output as gzip file
	params["local_gzipped_dst_file"] = local_dst_file + ".gz"
	with open(local_dst_file, 'rb') as f_in:
		with gzip.open(params["local_gzipped_dst_file"], 'wb') as f_out:
			f_out.writelines(f_in)
	#Remove unneeded files from local drive
	safely_remove_file(local_dst_file)
	safely_remove_file(local_src_file)
	#Push local gzipped dst_file to S3
	push_file_to_s3(params, "dst")
	#Remove local gzipped dst file
	safely_remove_file(params["local_gzipped_dst_file"])
	#Publish to the SNS topic
	post_SNS(dst_file_name + " successfully processed")
	# Remove
	push_file_to_s3(params, "err")
	sys.exit()

def run_from_command_line():
	"""Runs these commands if the module is invoked from the command line"""
	validate_configuration("config/daemon/schema.json",
		"config/daemon/file.json")
	params = initialize()
	process_single_input_file(params)

def run_meerkat_chunk(params, desc_queue, hyperparameters, cities):
	"""Run meerkat on a chunk of data"""
	# Run the Classifier
	consumer_threads = params.get("concurrency", 8)
	result_queue = queue.Queue()
	header = get_panel_header(params)
	output = []
	# Launch consumer threads
	for i in range(consumer_threads):
		new_consumer = FileConsumer(i, params, desc_queue,\
			result_queue, hyperparameters, cities)
		new_consumer.setDaemon(True)
		new_consumer.start()
	# Block thread execution on the consumer until all
	# items in the desc_queue have been processed
	desc_queue.join()
	# Dequeue results into dataframe
	while result_queue.qsize() > 0:
		row = result_queue.get()
		output.append(row)
		result_queue.task_done()
	# Block thread execution on the consumer until all
	# items in the result_queue have been processed
	result_queue.join()
	# Shutdown Loggers
	logging.shutdown()
	# Return a dataframe containing the results
	return pd.DataFrame(data=output, columns=header)

def load_hyperparameters(filepath):
	"""Attempts to load parameter key"""
	hyperparameters = None
	try:
		input_file = open(filepath, encoding='utf-8')
		hyperparameters = json.loads(input_file.read())
		input_file.close()
	except IOError:
		logging.error("%s not found, aborting.", filepath)
		sys.exit()
	return hyperparameters

def flush_errors(params, errors, dst_file_name, line_count):
	"""Takes all of the errors encountered and does the following:
		1.  Flushes individual error lines to a file.
		2.  Flushes a summary line with metrics about the error rate.
		3.  Writes the completed file out to the local host."""
	print("Flushing errors")
	my_options = params["my_producer_options"]
	dst_local_path = my_options["local_files"]["dst_path"]
	error_count = len(errors)
	# Set the name of the error file
	params["local_gzipped_err_file"] =\
		params["my_producer_options"]["local_files"]["err_path"]\
		+ dst_file_name + "." + params["location_pair"]["name"] + ".error.gz"
	# Make sure to clobber any existing error file first
	if "new_error_file" not in params:
		safely_remove_file(params["local_gzipped_err_file"])
		params["new_error_file"] = True
	# Write out a simple overall report as a summary
	error_msg = "Errors/Total Transactions {0}/{1}".format(error_count, line_count)
	write_error_file(params, error_msg)
	# Write all line errors out to a local gziped error file
	if error_count > 0:
		for error in errors:
			write_error_file(params, error)

def run_chunk(params, *argv):
	"""Run a single chunk from a dataframe_reader"""
	chunk, line_count, my_stderr, old_stderr = argv[:4]
	hyperparameters, cities, header, dst_file_name = argv[4:8]
	first_chunk, errors = argv[8:10]
	my_options = params["my_producer_options"]
	dst_local_path = my_options["local_files"]["dst_path"]
	# Save Errors
	line_count += chunk.shape[0]
	error_chunk = str.strip(my_stderr.getvalue())
	if len(error_chunk) > 0:
		errors += error_chunk.split('\n')
	sys.stderr = old_stderr
	# Clean Data
	chunk = clean_dataframe(params, chunk)
	# Load into Queue for Processing
	desc_queue, non_physical = convert_dataframe_to_queue(params, chunk)

	physical = None
	if not desc_queue.empty():
		# Classify Transaction Chunk
		logging.warning("Chunk contained physical transactions, using Meerkat")
		physical = run_meerkat_chunk(params, desc_queue, hyperparameters, cities)
	else:
		logging.warning("Chunk did not contain physical transactions, skipping Meerkat")
	# Combine Split Dataframes
	chunk = pd.concat([physical, non_physical])
	# Write
	if first_chunk:
		file_to_remove = dst_local_path + dst_file_name
		safely_remove_file(file_to_remove)
		logging.warning("Output Path: {0}".format(file_to_remove))
		chunk.to_csv(dst_local_path + dst_file_name, columns=header, sep="|",\
			mode="a", encoding="utf-8", index=False, index_label=False)
		first_chunk = False
	else:
		chunk.to_csv(dst_local_path + dst_file_name, header=False, columns=header,\
			sep="|", mode="a", encoding="utf-8", index=False, index_label=False)
	# Handle Errors
	sys.stderr = my_stderr = io.StringIO()
	return line_count, errors, first_chunk

def run_panel(params, dataframe_reader, dst_file_name):
	"""Process a single panel"""
	my_options = params["my_producer_options"]
	hyperparameters = load_hyperparameters(my_options["hyperparameters"])
	dst_local_path = my_options["local_files"]["dst_path"]
	remove_list = ["DESCRIPTION_UNMASKED", "GOOD_DESCRIPTION"]
	header = [x for x in get_panel_header(params) if x not in remove_list]
	cities = get_us_cities()
	line_count = 0
	first_chunk = True
	# Capture Errors
	errors = []
	# Save stderr context
	old_stderr = sys.stderr
	sys.stderr = my_stderr = io.StringIO()
	# Iterate through each chunk in the dataframe_reader
	for chunk in dataframe_reader:
		args = (chunk, line_count, my_stderr, old_stderr,
			hyperparameters, cities, header, dst_file_name, first_chunk,
			errors)
		line_count, errors, first_chunk = run_chunk(params, *args)
		logging.warning("{0} Lines processed.".format(line_count))
	#Restore stderr context
	sys.stderr = old_stderr
	# Flush errors to a file
	flush_errors(params, errors, dst_file_name, line_count)
	return dst_local_path + dst_file_name

def usage():
	"""Shows the user which parameters to send into the program."""
	result = "Usage: <location_pair_name> <file_name>"
	logging.error(result)
	return result

def validate_configuration(schema, config):
	"""Ensures that the correct parameters are supplied."""
	schema = json.load(open(schema))
	config = json.load(open(config))
	try:
		validate(config, schema)
	except ValidationError as validation_error:
		logging.error("Configuration file was invalid, aborting.")
		logging.error(validation_error.message)
		sys.exit()
	except SchemaError as schema_error:
		logging.error("Schema definition is invalid, aborting")
		logging.error(schema_error.message)
		sys.exit()
	logging.warning("Configuration schema is valid.")

def write_error_file(params, error_msg):
	"""Writes a gzipped file of errors to the local host."""
	logging.warning("Writing to error file: {0}".format(params["local_gzipped_err_file"]))
	if error_msg == "":
		return
	with gzip.open(params["local_gzipped_err_file"], "ab") as gzipped_output:
		gzipped_output.write(bytes(error_msg + "\n", 'UTF-8'))

if __name__ == "__main__":
	logging.basicConfig(format='%(asctime)s %(message)s',
		filename='/data/1/log/' + sys.argv[1] + '.' + sys.argv[2] + '.log', level=logging.WARNING)
	logging.warning("file_producer module activated.")
	run_from_command_line()
