#!/usr/local/bin/python3.3

"""This module loads and processes transactions in preparation for
analysis and search though our structured ElasticSearch merchant index.

Created on Dec 9, 2013
@author: J. Andrew Key
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3.3 -m meerkat [config_file]
# python3.3 -m meerkat config/users.json

#####################################################

import boto
import collections
import csv
import datetime
import gzip
import io
import json
import logging
import os
import pandas as pd
import queue
import re
import string
import sys

from boto.s3.connection import Location, S3Connection, Key
from jsonschema import validate

from meerkat.custom_exceptions import InvalidArguments, Misconfiguration
from meerkat.file_consumer import FileConsumer
from meerkat.classification.load import select_model
from meerkat.various_tools import (load_dict_list, safely_remove_file,\
	load_hyperparameters, safe_print)
from meerkat.various_tools import (split_csv, merge_split_files,\
	queue_to_list, string_cleanse, clean_bad_escapes)
from meerkat.various_tools import (get_panel_header, get_column_map,\
	get_new_columns, get_us_cities, post_SNS)
from meerkat.accuracy import test_accuracy, print_results, speed_tests
from meerkat.optimization import run_meerkat as test_meerkat
from meerkat.optimization import get_desc_queue as get_simple_queue
from meerkat.optimization import load_dataset

def clean_dataframe(params, df):
	"""Fix issues with current dataframe, like remapping, etc."""
	container = params["container"]
	column_remap = get_column_map(container)
	header = get_panel_header(container)
	new_columns = get_new_columns()
	# Rename and add columns
	df = df.rename(columns=column_remap)
	for column in new_columns:
		df[column] = ""
	# Reorder header
	df = df[header]
	return df

def connect_to_S3():
	"""Returns a connection to S3"""
	try:
		conn = boto.connect_s3()
	except boto.s3.connection.HostRequiredError:
		print("Error connecting to S3, check your credentials")
		sys.exit()
	return conn

def df_to_queue(params, df):
	"""Converts a dataframe to a queue for processing"""

	container = params["container"]
	classifier = select_model(container)
	classes = ["Non-Physical", "Physical", "ATM"]
	f = lambda x: classes[int(classifier(x["DESCRIPTION_UNMASKED"]))]
	desc_queue = queue.Queue()
	name_map = {"GOOD_DESCRIPTION" : "MERCHANT_NAME",\
		"MERCHANT_NAME" : "GOOD_DESCRIPTION"}

	# Classify transactions
	df['TRANSACTION_ORIGIN'] = df.apply(f, axis=1)
	gb = df.groupby('TRANSACTION_ORIGIN')
	groups = list(gb.groups)

	# Group into separate dataframes
	physical = gb.get_group("Physical") if "Physical" in groups else pd.DataFrame()
	atm = gb.get_group("ATM") if "ATM" in groups else pd.DataFrame()
	non_physical = gb.get_group("Non-Physical").rename(columns=name_map) if "Non-Physical" in groups else pd.DataFrame()

	#Return if there are no physical transactions
	if physical.empty:
		return desc_queue, non_physical

	# Roll ATM into physical
	physical = pd.concat([physical, atm])

	# Group by user
	users = physical.groupby('UNIQUE_MEM_ID')

	for user in users:
		user_trans = []
		for i, row in user[1].iterrows():
			user_trans.append(row.to_dict())
		desc_queue.put(user_trans)

	return desc_queue, non_physical

def get_container(filename):
	container = None
	if "bank" in filename.lower():
		container = "bank"
	elif "card" in filename.lower():
		container = "card"
	if container:
		logging.info("{0} discovered as container.".format(container))
		return container
	logging.error("Neither 'bank' nor 'card' present in file name, aborting.")
	#TODO Add a proper exception
	sys.exit()

def get_dataframe_reader(input_filename):
	"""Returns pandas dataframe reader for the input file."""
	return pd.read_csv(input_filename, na_filter=False, chunksize=5000, quoting=csv.QUOTE_NONE, encoding="utf-8", sep='|', error_bad_lines=False)

def gunzip_and_validate_file(filepath):
	"""Decompress a file, check for required fields on the first line."""
	logging.info("Gunzipping and validating {0}".format(filepath))
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
							sys.exit()
					first_line = False
				output_file.write(line)
	# Remove original file
	safely_remove_file(filepath)
	logging.info("{0} unzipped; header contains mandatory fields.".format(filename))
	return filename


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
			params["location_pair"] = item
			del params["location_pairs"]
			found = True
			break
	if not found:
		raise InvalidArguments(msg="Invalid 'location_pair' argument, aborting.", expr=None)
	logging.info("location_pair found in configuration file.")
	params["src_file"] = sys.argv[2]
	return params

def move_to_S3(params, bucket, s3_path, filepath):
	"""Moves a file to S3"""
	filename = os.path.basename(filepath)
	s3_path = s3_path + params["container"] + "/"
	key = Key(bucket)
	key.key = s3_path + filename
	bytes_written = key.set_contents_from_filename(filepath, encrypt_key=True, replace=True)
	safe_print("File written to: " + key.key)
	safe_print("{0} bytes written".format(bytes_written))
	safely_remove_file(filepath)

def pull_src_file_from_s3(params):
	#Grab relevant configuraiton parameters
	src_s3_location = params["location_pair"]["src_location"]
	location_pattern = re.compile("^([^/]+)/(.*)$")
	matches = location_pattern.search(src_s3_location)
	bucket_name = matches.group(1)
	directory = matches.group(2)
	src_file = params["src_file"]
	logging.info("S3 Bucket: {0}, S3 directory: {1}, Src file: {2}".format(bucket_name, directory, src_file))

	#Pull the src file from S3
	conn = connect_to_S3()
	bucket = conn.get_bucket(bucket_name, Location.USWest2)
	listing = bucket.list(prefix=directory+src_file)
	s3_key = None
	for item in listing:
		s3_key = item
	params["local_src_path"] = params["producer_options"]["producer_default"]["local_files"]["src_path"]
	local_src_file = params["local_src_path"] + src_file
	s3_key.get_contents_to_filename(local_src_file)
	logging.info("Src file pulled from S3")
	return local_src_file

def push_dst_file_to_s3(params):
#def move_to_S3(params, bucket, s3_path, filepath):
	"""Moves a file to S3"""
	dst_s3_location = params["location_pair"]["dst_location"]
	location_pattern = re.compile("^([^/]+)/(.*)$")
	matches = location_pattern.search(dst_s3_location)
	bucket_name = matches.group(1)
	directory = matches.group(2)
	dst_file = params["src_file"]
	#Push the dst file to S3
	conn = connect_to_S3()
	bucket = conn.get_bucket(bucket_name, Location.USWest2)
	key = Key(bucket)
	key.key = directory + dst_file
	bytes_written = key.set_contents_from_filename(params["local_gzipped_dst_file"], encrypt_key=True, replace=True)
	logging.info("{0} pushed to S3, {1} bytes written.".format(dst_file, bytes_written))

def push_err_file_to_s3(params):
	"""Unimplemented."""
	#TODO: Implement
	pass

def run(params):
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
	reader = get_dataframe_reader(local_src_file)
	logging.info("Dataframe reader loaded.")
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
	push_dst_file_to_s3(params)
	#Remove local gzipped dst file
	safely_remove_file(params["local_gzipped_dst_file"])
	#Publish to the SNS topic
	post_SNS(dst_file_name + " successfully processed")
	# Remove
	push_err_file_to_s3(params)
	sys.exit()

def run_from_command_line(command_line_arguments):
	"""Runs these commands if the module is invoked from the command line"""
	params = initialize()
	validate_configuration()
	run(params)

def run_meerkat_chunk(params, desc_queue, hyperparameters, cities):
	"""Run meerkat on a chunk of data"""
	# Run the Classifier
	consumer_threads = params.get("concurrency", 8)
	result_queue = queue.Queue()
	header = get_panel_header(params["container"])
	output = []

	for i in range(consumer_threads):
		new_consumer = FileConsumer(i, params, desc_queue,\
			result_queue, hyperparameters, cities)
		new_consumer.setDaemon(True)
		new_consumer.start()

	desc_queue.join()

	# Dequeue results into dataframe
	while result_queue.qsize() > 0:
		row = result_queue.get()
		output.append(row)
		result_queue.task_done()

	result_queue.join()

	# Shutdown Loggers
	logging.shutdown()

	return pd.DataFrame(data=output, columns=header)

def run_panel(params, reader, dst_file_name):
	"""Process a single panel"""
	my_options = params["producer_options"]["producer_default"]
	#Lots of adapting to existing code, sheesh
	params["input"] = {}
	params["input"]["hyperparameters"] = my_options["hyperparameters"]
	params["elasticsearch"] = my_options["elasticsearch"]
	params["logging"] = my_options["logging"]
	params["output"] = my_options["output"]
	params["output"]["results"] = {}
	params["output"]["results"]["fields"] = []
	params["output"]["results"]["labels"] = []
	for field, label in my_options["field_mappings"]:
		params["output"]["results"]["fields"].append(field)
		params["output"]["results"]["labels"].append(label)
	params["mode"] = "production"
	#Getting on with it
	hyperparameters = load_hyperparameters(params)
	#dst_local_path = params["input"]["S3"]["dst_local_path"]
	dst_local_path = my_options["local_files"]["dst_path"]
	header = get_panel_header(params["container"])[0:-2]
	cities = get_us_cities()
	line_count = 0
	first_chunk = True

	# Capture Errors
	errors = []
	old_stderr = sys.stderr
	sys.stderr = mystderr = io.StringIO()

	for chunk in reader:

		# Save Errors
		line_count += chunk.shape[0]
		error_chunk = str.strip(mystderr.getvalue())
		if len(error_chunk) > 0:
			errors += error_chunk.split('\n')
		sys.stderr = old_stderr

		# Clean Data
		chunk = clean_dataframe(params, chunk)

		# Load into Queue for Processing
		desc_queue, non_physical = df_to_queue(params, chunk)

		physical = None
		if not desc_queue.empty():
			# Classify Transaction Chunk
			logging.info("Chunk contained physical transactions, using Meerkat")
			physical = run_meerkat_chunk(params, desc_queue, hyperparameters, cities)
		else:
			logging.info("Chunk did not contain physical transactions, skipping Meerkat")
		# Combine Split Dataframes
		chunk = pd.concat([physical, non_physical])

		# Write
		if first_chunk:
			safely_remove_file(dst_local_path + dst_file_name)
			safe_print("Output Path: " + dst_local_path + dst_file_name)
			chunk.to_csv(dst_local_path + dst_file_name, columns=header, sep="|",\
				mode="a", encoding="utf-8", index=False, index_label=False)
			first_chunk = False
		else:
			chunk.to_csv(dst_local_path + dst_file_name, header=False, columns=header,\
				sep="|", mode="a", encoding="utf-8", index=False, index_label=False)

		# Handle Errors
		sys.stderr = mystderr = io.StringIO()

	sys.stderr = old_stderr

	# Write Errors
	error_count = len(errors)
	if error_count > 0:
		for error in errors:
			write_error_file(dst_local_path, dst_file_name, error)
		error_summary = [str(line_count), str(error_count), str(1.0 * (error_count / line_count))]
		error_msg = "Total line count: {}\nTotal error count: {}\n Success Ratio: {}"
		error_msg = error_msg.format(*error_summary)
		write_error_file(dst_local_path, dst_file_name, error_msg)

	return dst_local_path + dst_file_name

def usage():
	"""Shows the user which parameters to send into the program."""
	result = "Usage:\n\t<location_pair_name> <file_name>"
	#gpanel2_bank 20150205.007.012_GPANEL2_BANK.txt.gz
	logging.error(result)
	return result

def validate_configuration():
	"""Ensures that the correct parameters are supplied."""
	schema_file = open("config/daemon/schema.json")
	config_file = open("config/daemon/file.json")

	schema = json.load(schema_file)
	config = json.load(config_file)

	result = validate(config, schema)
	logging.info("Configuration schema is valid.")


def write_error_file(path, filename, error_msg):
	with gzip.open(path + filename + ".error.gz", "ab") as gzipped_output:
		if error_msg != "":
			gzipped_output.write(bytes(error_msg + "\n", 'UTF-8'))

if __name__ == "__main__":
	logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/file_producer.log', \
		level=logging.INFO)
	logging.info("file_producer module activated.")
	run_from_command_line(sys.argv)
