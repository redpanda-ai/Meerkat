import argparse
import csv
import json
import logging
import os


import numpy as np
import pandas as pd

from boto.s3.connection import Location
from boto import connect_s3
from plumbum import local, NOHUP

def dict_2_json(obj, filename):
	"""Saves a dict as a json file"""
	logging.info("Generating JSON.")
	with open(filename, 'w') as output_file:
		json.dump(obj, output_file, indent=4)

def cap_first_letter(label):
	"""Make sure the first letter of each word is capitalized"""
	temp = label.split()
	for i in range(len(temp)):
		if temp[i].lower() in ['by', 'with', 'or', 'at', 'in']:
			temp[i] = temp[i].lower()
		else:
			temp[i] = temp[i][0].upper() + temp[i][1:]
	return ' '.join(word for word in temp)

def pull_from_s3(**kwargs):
	"""Pulls the contents of an S3 directory into a local file, returning
	the first file"""
	bucket_name, prefix = kwargs["bucket"], kwargs["prefix"]
	logging.info("Scanning S3 at {0}".format(bucket_name + "/" + prefix))
	conn = connect_s3()
	bucket = conn.get_bucket(bucket_name, Location.USWest2)
	listing = bucket.list(prefix=prefix)

	my_filter = kwargs["my_filter"]
	logging.info("Filtering S3 objects by '{0}' extension".format(my_filter))
	s3_object_list = [
		s3_object
		for s3_object in listing
		if s3_object.key[-len(my_filter):] == my_filter
	]
	get_filename = lambda x: kwargs["input_path"] + x.key[x.key.rfind("/")+1:]
	local_files = []
	# Collect all files at the S3 location with the correct extension and write
	# Them to a local file
	
	for s3_object in s3_object_list:
		local_file = get_filename(s3_object)
		logging.info("Found the following file: {0}".format(local_file))
		s3_object.get_contents_to_filename(local_file)
		local_files.append(local_file)

	# However we only wish to process the first one
	if local_files:
		return local_files.pop()
	# Or give an informative error, if we don't have any
	else:
		raise Exception("Cannot proceed, there must be at least one file at the"
			" S3 location provided.")

def load(**kwargs):
	"""Load the CSV into a pandas data frame"""
	filename, debit_or_credit = kwargs["input_file"], kwargs["debit_or_credit"]
	logging.info("Loading csv file and slicing by '{0}' ".format(debit_or_credit))
	df = pd.read_csv(filename, quoting=csv.QUOTE_NONE, na_filter=False,
		encoding="utf-8", sep='|', error_bad_lines=False, low_memory=False)
	df['UNIQUE_TRANSACTION_ID'] = df.index
	df['LEDGER_ENTRY'] = df['LEDGER_ENTRY'].str.lower()
	grouped = df.groupby('LEDGER_ENTRY', as_index=False)
	groups = dict(list(grouped))
	df = groups[debit_or_credit]
	df["PROPOSED_SUBTYPE"] = df["PROPOSED_SUBTYPE"].str.strip()
	df['PROPOSED_SUBTYPE'] = df['PROPOSED_SUBTYPE'].apply(cap_first_letter)
	class_names = df["PROPOSED_SUBTYPE"].value_counts().index.tolist()
	return df, class_names

def get_label_map(**kwargs):
	"""Generates a label map (class_name: label number)."""
	logging.info("Generating label map")
	class_names = kwargs["class_names"]
	# Create a label map
	label_numbers = list(range(1, (len(class_names) + 1)))
	label_map = dict(zip(sorted(class_names), label_numbers))
	# Map Numbers
	my_lambda = lambda x: label_map[x["PROPOSED_SUBTYPE"]]
	df = kwargs["df"]
	df['LABEL'] = df.apply(my_lambda, axis=1)
	return label_map

def get_test_and_train_dataframes(**kwargs):
	"""Produce (rich and poor) X (test and train) dataframes"""
	logging.info("Building test and train dataframes")
	df_rich = kwargs["df"]
	df_poor = df_rich[['LABEL', 'DESCRIPTION_UNMASKED']]
	msk = np.random.rand(len(df_poor)) < 0.90
	return {
		"df_poor_train" : df_poor[msk],
		"df_poor_test" : df_poor[~msk],
		"df_rich_train" : df_rich[msk],
		"df_test" : df_rich[~msk]
	}

def get_json_and_csv_files(**kwargs):
	"""This function generates CSV and JSON files"""
	prefix = kwargs["output_path"] + kwargs["bank_or_card"] + "." + kwargs["debit_or_credit"] + "."
	logging.info("Prefix is : {0}".format(prefix))
	#set file names
	files = {
		"map_file" : prefix + "map.json",
		"test_rich" : prefix + "test.rich.csv",
		"train_rich" : prefix + "train.rich.csv",
		"test_poor" : prefix + "test.poor.csv",
		"train_poor" : prefix + "train.poor.csv"
	}
	#Write the JSON file
	dict_2_json(kwargs["label_map"], files["map_file"])
	#Write the rich CSVs
	rich_kwargs = {"index" : False, "sep" : "|"}
	kwargs["df_test"].to_csv(files["test_rich"], **rich_kwargs)
	kwargs["df_rich_train"].to_csv(files["train_rich"], **rich_kwargs)
	#Write the poor CSVs
	poor_kwargs = {"cols" : ["LABEL", "DESCRIPTION_UNMASKED"], "header": False,
		"index" : False, "index_label": False}
	kwargs["df_poor_test"].to_csv(files["test_poor"], **poor_kwargs)
	kwargs["df_poor_train"].to_csv(files["train_poor"], **poor_kwargs)
	#Return file names
	return files

def fill_description_unmasked(row):
	"""Ensures that blank values for DESCRIPTION_UNMASKED are always filled."""
	if row["DESCRIPTION_UNMASKED"] == "":
		return row["DESCRIPTION"]
	return row["DESCRIPTION_UNMASKED"]

def slice_into_dataframes(**kwargs):
	"""Slice into test and train dataframs, make a label map, and produce 
	CSV files."""
	# Create an output directory if it does not exist
	os.makedirs(kwargs["output_path"], exist_ok=True)
	# Load data frame and class names
	df, class_names = load(input_file=kwargs["input_file"], debit_or_credit=kwargs["debit_or_credit"])
	# Generate a mapping (class_name: label_number)
	label_map = get_label_map(df=df, class_names=class_names)
	# Reverse the mapping (label_number: class_name)
	kwargs["label_map"] = dict(zip(label_map.values(), label_map.keys()))
	# Clean the "DESCRIPTION_UNMASKED" values within the dataframe
	df["DESCRIPTION_UNMASKED"] = df.apply(fill_description_unmasked, axis=1)
	kwargs["df"] = df
	# Make Test and Train data frames
	kwargs.update(get_test_and_train_dataframes(**kwargs))
	# Generate the output files (CSV and JSON) and return the file handles
	kwargs.update(get_json_and_csv_files(**kwargs))
	#logging.info("The kwargs dictionary contains: \n{0}".format(kwargs))
	return kwargs["train_poor"], kwargs["test_poor"], len(class_names)

def parse_arguments():
	"""This function parses arguments from our command line."""
	parser = argparse.ArgumentParser("stream")
	parser.add_argument("-d", "--debug", help="log at DEBUG level",
		action="store_true")
	parser.add_argument("-v", "--info", help="log at INFO level",
		action="store_true")

	parser.add_argument("output_dir", help="Where do you want to write out all of your files?")
	parser.add_argument("card_or_bank", help="Whether we are processing card or bank transactions.")
	parser.add_argument("debit_or_credit",
		help="What kind of transactions do you wanna process, debit or credit?")

	args = parser.parse_args()
	if args.debug:
		logging.basicConfig(level=logging.DEBUG)
	elif args.info:
		logging.basicConfig(level=logging.INFO)
	return args

def convert_csv_to_torch_7_binaries(input_file):
	"""Use plumbum to convert CSV files to torch 7 binaries."""
	output_file = input_file[:-4] + ".t7b"
	logging.info("Converting {0} to {1}.".format(input_file, output_file))
	# Remove the output_file
	if os.path.isfile(output_file):
		os.remove(output_file)
	command = local["qlua"]["meerkat/classification/lua/csv2t7b.lua"]\
		["-input"][input_file]["-output"][output_file]
	result = command()
	logging.info("The result is {0}".format(result))
	return output_file

def create_new_configuration_file(num_of_classes, output_path, train_file, test_file):
	"""This function generates a new config.lua file based off of a template."""
	logging.info("Generate a new configuration file with the correct number of classes.")
	logging.info("Output_path = {0}".format(output_path))
	output_dir_len = len(output_path)
	train_file = train_file[output_dir_len:]
	test_file = test_file[output_dir_len:]
	command = \
		local["sed"]["s:\./train\.t7b:" + train_file + ":"]["meerkat/classification/lua/config.lua"] \
		| local["sed"]["s:\./test\.t7b:" + test_file + ":"] \
		 > output_path + "config.lua"
	command()

def copy_file(input_file, directory):
	"""This function moves uses Linux's 'cp' command to copy files on the local host"""
	logging.info("Copy the file {0} to directory: {1}".format(input_file, directory))
	local["cp"][input_file][directory]()

def execute_main_lua(output_path, input_file):
	"""This function executes the qlua command and launches the lua script in the background."""
	logging.info("Executing main.lua in background.")
	with local.cwd(output_path):
		(local["qlua"][input_file]) & NOHUP
	logging.info("It's running.")

if __name__ == "__main__":
	logging.error("Sorry, this module is a useful library of useful "
		"functions you can import into your code, you should not "
		"execute it from the command line.")
