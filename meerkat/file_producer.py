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
from meerkat.consumer import Consumer
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

def initialize():
	"""Validates the command line arguments."""
	input_file, params = None, None
	if len(sys.argv) != 2:
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
	return params

def usage():
	"""Shows the user which parameters to send into the program."""
	result = "Usage:\n\t<location_name>"
	logging.error(result)
	return result

def validate_configuration():
	schema_file = open("config/daemon/schema.json")
	config_file = open("config/daemon/file.json")

	schema = json.load(schema_file)
	config = json.load(config_file)

	result = validate(config, schema)
	logging.info("Configuration schema is valid.")

def validate_params(params):
	"""Ensures that the correct parameters are supplied."""

	mandatory_keys = ["elasticsearch", "concurrency", "input", "logging", "mode"]

	# Ensure Mandatory Keys are in Config
	for key in mandatory_keys:
		if key not in params:
			raise Misconfiguration(msg="Misconfiguration: missing key, '" + key + "'", expr=None)

	if params["concurrency"] <= 0:
		raise Misconfiguration(msg="Misconfiguration: 'concurrency' must be a positive integer", expr=None)

	if "hyperparameters" not in params["input"]:
		params["input"]["hyperparameters"] = "config/hyperparameters/default.json"

	# Ensure Test Mode Requirements
	if params["mode"] == "test":
		if params.get("verification_source", "") == "":
			raise Misconfiguration(msg="Please provide verification_source to run in test mode", expr=None)

	# Ensure Other Various Parameters Available
	if "index" not in params["elasticsearch"]:
		raise Misconfiguration(msg="Misconfiguration: missing key, 'elasticsearch.index'", expr=None)
	if "type" not in params["elasticsearch"]:
		raise Misconfiguration(msg="Misconfiguration: missing key, 'elasticsearch.type'", expr=None)
	if "cluster_nodes" not in params["elasticsearch"]:
		raise Misconfiguration(msg="Misconfiguration: missing key, 'elasticsearch.cluster_nodes'", expr=None)
	if "path" not in params["logging"]:
		raise Misconfiguration(msg="Misconfiguration: missing key, 'logging.path'", expr=None)
	if "filename" not in params["input"] and "S3" not in params["input"]:
		raise Misconfiguration(msg="Misconfiguration: missing key, 'input.filename' or 'input.S3'", expr=None)
	if "encoding" not in params["input"]:
		raise Misconfiguration(msg="Misconfiguration: missing key, 'input.encoding'", expr=None)

	return True

def	hms_to_seconds(t):
	h, m, s = [i.lstrip("0") if len(i.lstrip("0")) != 0 else 0 for i in t.split(':')]
	print ("H: {0}, M: {1}, S: {2}".format(h, m, s))
	return 3600 * float(h) + 60 * float(m) + float(s)

def get_S3_buckets(S3_params, conn):
	"""Gets src, dst, and error S3 buckets"""

	bucket_filter = S3_params["filter"]
	src_bucket_name = S3_params["src_bucket_name"]
	src_bucket = conn.get_bucket(src_bucket_name, Location.USWest2)
	dst_bucket_name = S3_params["dst_bucket_name"]
	dst_bucket = conn.get_bucket(dst_bucket_name, Location.USWest2)
	error_bucket_name = S3_params["error_bucket_name"]
	error_bucket = conn.get_bucket(error_bucket_name, Location.USWest2)

	return src_bucket, dst_bucket, error_bucket

def get_S3_regex(S3_params, container, folders):
	"""Get a regex for dealing with S3 buckets"""

	return re.compile(folders + container + "/(" + S3_params["filter"] + "[^/]+)")

def get_completed_files(bucket, path_regex):
	"""Get a list of files that have been processed"""

	completed = {}

	for j in bucket.list():
		if path_regex.search(j.key):
			completed[path_regex.search(j.key).group(1)] = j.size

	return completed

def get_pending_files(bucket, path_regex, completed):
	"""Get a list of files that need to be processed"""

	pending = []

	for k in bucket.list():
		if path_regex.search(k.key):
			file_name = path_regex.search(k.key).group(1)
			if file_name in completed:
				ratio = float(k.size) / completed[file_name]
				if ratio >= 1.8:
					print("Completed Size, Source Size, Ratio: {0}, {1}, {2:.2f}".format(completed[file_name], k.size, ratio))
					print("Re-running {0}".format(file_name))
					pending.append(k)
			elif k.size > 0:
				pending.append(k)

	return pending, completed

def identify_container(params):
	"""Determines whether transactions are bank or card"""

	container = params.get("container", "")
	filename = params["input"]["filename"]

	if "bank" in filename.lower():
		return "bank"
	elif "card" in filename.lower():
		return "card"
	elif container == "bank" or container == "card":
		return container
	else:
		print('Please designate whether this is bank or card in params["container"]')
		sys.exit()

def production_run(params):
	"""Runs Meerkat in production mode"""

	conn = connect_to_S3()
	container = params["container"]
	S3_params =  params["input"]["S3"]

	# Get Buckets
	src_bucket, dst_bucket, error_bucket = get_S3_buckets(S3_params, conn)
	src_s3_path_regex = get_S3_regex(S3_params, container, S3_params["src_folders"])
	dst_s3_path_regex = get_S3_regex(S3_params, container, S3_params["dst_folders"])

	# Sort By Completion Status
	completed = get_completed_files(dst_bucket, dst_s3_path_regex)
	pending, completed = get_pending_files(src_bucket, src_s3_path_regex, completed)

	# Exit if Nothing to Process
	if not pending:
		print("Everything is up-to-date.")
		sys.exit()

	# Process in Reverse Chronological Order
	pending.reverse()

	# Process Files
	for item in pending:

		src_file_name = src_s3_path_regex.search(item.key).group(1)
		safe_print("Processing file: " + src_file_name)

		# Copy from S3
		item.get_contents_to_filename(S3_params["src_local_path"] + src_file_name)
		params["input"]["filename"] = S3_params["src_local_path"] + src_file_name

		# Remove Troublesome Escapes and decompress
		result = clean_bad_escapes(params["input"]["filename"])

		# File must have header
		if not result:
			filename = os.path.splitext(src_file_name)[0]
			error_filepath = S3_params["error_local_path"] + filename + ".error.gz"
			write_error_file(S3_params["error_local_path"], filename, "No header found in source file")
			move_to_S3(params, error_bucket, S3_params["error_s3_path"], error_filepath)
			safely_remove_file(error_filepath)
			safely_remove_file(params["input"]["filename"])
			continue

		# Save Details and Continue
		dst_file_name = src_file_name = result
		params["input"]["filename"] = S3_params["src_local_path"] + src_file_name

		# Load into Dataframe
		container = identify_container(params)
		params["container"] = container
		reader = load_dataframe(params)

		# Process With Meerkat
		local_dst_filepath = run_panel(params, reader, dst_file_name)

		# Gzip and Remove
		with open(local_dst_filepath, 'rb') as f_in:
			with gzip.open(local_dst_filepath + ".gz", 'wb') as f_out:
				f_out.writelines(f_in)

		safely_remove_file(local_dst_filepath)
		safely_remove_file(S3_params["src_local_path"] + src_file_name)

		# Push to S3
		error_filepath = S3_params["error_local_path"] + dst_file_name + ".error.gz"
		move_to_S3(params, dst_bucket, S3_params["dst_s3_path"], local_dst_filepath + ".gz")
		post_SNS(dst_file_name + " successfully processed")
		#Ensure that error file exists before pushing to S3
		if os.path.isfile(error_filepath):
			move_to_S3(params, error_bucket, S3_params["error_s3_path"], error_filepath)
			safely_remove_file(error_filepath)

def run_panel(params, reader, dst_file_name):
	"""Process a single panel"""

	hyperparameters = load_hyperparameters(params)
	dst_local_path = params["input"]["S3"]["dst_local_path"]
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
			physical = run_meerkat_chunk(params, desc_queue, hyperparameters, cities)

		# Combine Split Dataframes
		chunk = pd.concat([physical, non_physical])

		# Write 	
		if first_chunk:
			safely_remove_file(dst_local_path + dst_file_name)
			safe_print("Output Path: " + dst_local_path + dst_file_name)
			chunk.to_csv(dst_local_path + dst_file_name, columns=header, sep="|", mode="a", encoding="utf-8", index=False, index_label=False)
			first_chunk = False
		else:
			chunk.to_csv(dst_local_path + dst_file_name, header=False, columns=header, sep="|", mode="a", encoding="utf-8", index=False, index_label=False)

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

def write_error_file(path, filename, error_msg):
	with gzip.open(path + filename + ".error.gz", "ab") as gzipped_output:
		if error_msg != "":
			gzipped_output.write(bytes(error_msg + "\n", 'UTF-8'))

def run_meerkat_chunk(params, desc_queue, hyperparameters, cities):
	"""Run meerkat on a chunk of data"""

	# Run the Classifier
	consumer_threads = params.get("concurrency", 8)
	result_queue = queue.Queue()
	header = get_panel_header(params["container"])
	output = []

	for i in range(consumer_threads):
		new_consumer = Consumer(i, params, desc_queue,\
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

def df_to_queue(params, df):
	"""Converts a dataframe to a queue for processing"""

	container = params["container"]
	classifier = select_model(container)
	classes = ["Non-Physical", "Physical", "ATM"]
	f = lambda x: classes[int(classifier(x["DESCRIPTION_UNMASKED"]))]
	desc_queue = queue.Queue()
	name_map = {"GOOD_DESCRIPTION" : "MERCHANT_NAME", "MERCHANT_NAME" : "GOOD_DESCRIPTION"}

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

def clean_dataframe(params, df):
	"""Fix issues with current dataframe"""

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

def load_dataframe(params):
	"""Loads file into a pandas dataframe"""

	# Read file into dataframe
	reader = pd.read_csv(params["input"]["filename"], na_filter=False, chunksize=5000, quoting=csv.QUOTE_NONE, encoding="utf-8", sep='|', error_bad_lines=False)

	return reader

def mode_switch(params):
	"""Switches mode between, single file / s3 bucket mode"""

	mode = params.get("mode", "")

	if mode == "test" or mode == "interactive":
		params["verification_source"] = "data/misc/ground_truth_card.txt"
		params["container"] = "card"
		test_training_data(params)
		params["verification_source"] = "data/misc/ground_truth_bank.txt"
		params["container"] = "card"
		test_training_data(params)
		per_merchant_tests(params)
	elif mode == "production":
		production_run(params)
	else:
		logging.critical("Please provide a verification_source for testing or a s3 bucket for "\
			"procesing. Terminating")
		sys.exit()

def test_training_data(params):
	"""An easy way to test the accuracy of a small set
	provided a set of hyperparameters"""

	safe_print("Testing sample: " + params["verification_source"])
	hyperparameters = load_hyperparameters(params)
	dataset = load_dataset(params)
	desc_queue = get_simple_queue(dataset)
	test_meerkat(params, desc_queue, hyperparameters)

def per_merchant_tests(params):
	"""Run tests on a directory of Merchant Samples"""

	dir_name = "data/misc/Merchant Samples/"
	params["label_key"] = "MERCHANT_NAME"
	merchant_files = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if f.endswith(".txt")]

	for sample in merchant_files:
		params["verification_source"] = sample
		test_training_data(params)

def connect_to_S3():
	"""Returns a connection to S3"""

	try:
		conn = boto.connect_s3()
	except boto.s3.connection.HostRequiredError:
		print("Error connecting to S3, check your credentials")
		sys.exit()

	return conn

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

def run_from_command_line(command_line_arguments):
	"""Runs these commands if the module is invoked from the command line"""

	params = initialize()
	logging.info(params)
	#sys.exit()

	validate_configuration()
	sys.exit()
	validate_params(params)
	mode_switch(params)

if __name__ == "__main__":
	logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/file_producer.log', \
		level=logging.INFO)
	logging.info("file_producer module activated.")
	run_from_command_line(sys.argv)
