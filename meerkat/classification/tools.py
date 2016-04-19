"""The place where you put frequently used functions"""

import csv
import datetime
import json
import logging
import os
import sys
import tarfile

import numpy as np
import pandas as pd

from boto.s3.key import Key
from boto.s3.connection import Location
from boto import connect_s3
from plumbum import local
from meerkat.various_tools import load_piped_dataframe

def check_new_input_file(**s3_params):
	"""Check the existence of a new input.tar.gz file"""
	prefix = s3_params["prefix"]
	prefix = prefix + '/' * (prefix[-1] != '/')

	bucket = connect_s3().get_bucket(s3_params["bucket"], Location.USWest2)
	listing_version = bucket.list(prefix=prefix, delimiter='/')

	version_object_list = [
		version_object
		for version_object in listing_version
	]

	version_dir_list = []
	for i in range(len(version_object_list)):
		full_name = version_object_list[i].name
		if full_name.endswith("/"):
			dir_name = full_name[full_name.rfind("/", 0, len(full_name) - 1)+1:len(full_name)-1]
			if dir_name.isdigit():
				version_dir_list.append(dir_name)

	newest_version = sorted(version_dir_list, reverse=True)[0]
	newest_version_dir = prefix + newest_version
	logging.info("The newest direcory is: {0}".format(newest_version_dir))
	listing_tar_gz = bucket.list(prefix=newest_version_dir)

	tar_gz_object_list = [
		tar_gz_object
		for tar_gz_object in listing_tar_gz
	]

	tar_gz_file_list = []
	for i in range(len(tar_gz_object_list)):
		full_name = tar_gz_object_list[i].name
		tar_gz_file_name = full_name[full_name.rfind("/")+1:]
		tar_gz_file_list.append(tar_gz_file_name)

	if "input.tar.gz" not in tar_gz_file_list:
		logging.critical("input.tar.gz doesn't exist in {0}".format(newest_version_dir))
		sys.exit()
	elif "preprocessed.tar.gz" not in tar_gz_file_list:
		return True, newest_version_dir, newest_version
	else:
		return False, newest_version_dir, newest_version

def check_file_exist_in_s3(file_name, **s3_params):
	"""Check if a file exist in s3"""
	prefix = kwargs["prefix"]
	prefix = prefix + '/' * (prefix[-1] != '/')

	bucket = connect_s3().get_bucket(s3_params["bucket"], Location.USWest2)
	listing = bucket.list(prefix=prefix, delimiter='/')

	file_name = kwargs.get("file_name", '')
	s3_object_list = [
		s3_object
		for s3_object in listing
			if s3_object.key.endswith(file_name)
	]
	if len(s3_object_list) == 0:
		raise Exception('Unable to find {0}'.format(file_name))
	return True

def make_tarfile(output_filename, source_dir):
	"""Makes a gzipped tarball"""
	with tarfile.open(output_filename, "w:gz") as tar:
		reduced_name = os.path.basename(source_dir)[len(source_dir):]
		tar.add(source_dir, arcname=reduced_name)

def get_new_maint7b(directory, file_list):
	"""Get the latest t7b file under directory."""
	print("Get the latest main_*.t7b file")
	for i in os.listdir(directory):
		if i.startswith('main_') and i not in file_list:
			file_list.append(i)
			return i

def get_best_error_rate(erasDict):
	"""Get the best error rate among different eras"""
	bestErrorRate = 1.0
	bestEraNumber = 1

	df = pd.DataFrame.from_dict(erasDict, orient="index")
	bestErrorRate = df.min().values[0]
	bestEraNumber = df.idxmin().values[0]

	return bestErrorRate, bestEraNumber

def get_utc_iso_timestamp():
	"""Returns a 16 digit ISO timestamp, accurate to the second that is suitable for S3
		Example: "20160403164944" (April 3, 2016, 4:49:44 PM UTC) """
	return datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")

def push_file_to_s3(source_path, bucket_name, object_prefix):
	"""Pushes an object to S3"""
	conn = connect_s3()
	bucket = conn.get_bucket(bucket_name, Location.USWest2)
	filename = os.path.basename(source_path)
	key = Key(bucket)
	key.key = object_prefix + filename
	key.set_contents_from_filename(source_path)

def zip_cnn_stats_dir(file1, file2):
	"""Copy files to Best_CNN_Statics directory and zip it"""
	local["mkdir"]["Best_CNN_Statics"]()
	local["cp"][file1]["Best_CNN_Statics"]()
	local["cp"][file2]["Best_CNN_Statics"]()
	local["tar"]["-zcvf"]["Best_CNN_Statics.tar.gz"]["Best_CNN_Statics"]()

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

	extension = kwargs["extension"]
	logging.info("Filtering S3 objects by '{0}' extension".format(extension))
	file_name = kwargs.get("file_name", '')
	if file_name == '':
		s3_object_list = [
			s3_object
			for s3_object in listing
			if s3_object.key.endswith(extension)
			]
		if len(s3_object_list) != 1:
			raise Exception('There does not exist a unique {0} file under {1}.\
				Please specifiy a file name.'\
				.format(extension, prefix))
	else:
		s3_object_list = [
			s3_object
			for s3_object in listing
			if s3_object.key.endswith(file_name)
			]
		if len(s3_object_list) == 0:
			raise Exception('Unable to find {0}'.format(file_name))

	get_filename = lambda x: kwargs["save_path"] + x.key[x.key.rfind("/")+1:]
	# local_files = []
	# Collect all files at the S3 location with the correct extension and write
	# Them to a local file
	
	# for s3_object in s3_object_list:
	local_file = get_filename(s3_object_list[0])
	logging.info("Found the following file: {0}".format(local_file))
	s3_object_list[0].get_contents_to_filename(local_file)

	return local_file
	# local_files.append(local_file)
# pylint:disable=pointless-string-statement
"""
	# However we only wish to process the first one
	if local_files:
		return local_files.pop()
	# Or give an informative error, if we don't have any
	else:
		raise Exception("Cannot proceed, there must be at least one file at the"
			" S3 location provided.")
"""

def load(**kwargs):
	"""Load the CSV into a pandas data frame"""
	filename, credit_or_debit = kwargs["input_file"], kwargs["credit_or_debit"]
	logging.info("Loading csv file and slicing by '{0}' ".format(credit_or_debit))
	df = pd.read_csv(filename, quoting=csv.QUOTE_NONE, na_filter=False,
		encoding="utf-8", sep='|', error_bad_lines=False, low_memory=False)
	df['UNIQUE_TRANSACTION_ID'] = df.index
	df['LEDGER_ENTRY'] = df['LEDGER_ENTRY'].str.lower()
	grouped = df.groupby('LEDGER_ENTRY', as_index=False)
	groups = dict(list(grouped))
	df = groups[credit_or_debit]
	df["PROPOSED_SUBTYPE"] = df["PROPOSED_SUBTYPE"].str.strip()
	df['PROPOSED_SUBTYPE'] = df['PROPOSED_SUBTYPE'].apply(cap_first_letter)
	class_names = df["PROPOSED_SUBTYPE"].value_counts().index.tolist()
	return df, class_names

def get_label_map(class_names):
	"""Generates a label map (class_name: label number)."""
	logging.info("Generating label map")
	# Create a label map
	label_numbers = list(range(1, (len(class_names) + 1)))
	label_map = dict(zip(sorted(class_names), label_numbers))
	return label_map

def show_label_stat(results, train_or_test, label='LABEL'):
	"""Print count of each label"""
	key = 'df_poor_' + train_or_test
	print('Label counts for {0}ing set:'.format(train_or_test))
	temp = results[key][label].value_counts()
	temp.index = temp.index.astype(int)
	temp = temp.sort_index()
	pd.set_option('display.max_rows', len(temp))
	# print(temp)
	print("There are {0} classes".format(len(temp)))
	pd.reset_option('display.max_rows')

def get_test_and_train_dataframes(df_rich, train_size=0.90):
	"""Produce (rich and poor) X (test and train) dataframes"""
	logging.info("Building test and train dataframes")
	df_poor = df_rich[['LABEL', 'DESCRIPTION_UNMASKED']]
	msk = np.random.rand(len(df_poor)) < train_size
	results = {
		"df_poor_train" : df_poor[msk],
		"df_poor_test" : df_poor[~msk],
		# "df_rich_test" : df_rich[~msk],
		# "df_rich_train" : df_rich[msk]
	}
	show_label_stat(results, 'train')
	show_label_stat(results, 'test')
	return results

def get_csv_files(**kwargs):
	"""This function generates CSV and JSON files, returns paths of the files"""
	prefix = kwargs["output_path"] +  kwargs["merchant_or_subtype"] + '_' + \
		kwargs["bank_or_card"] + "_" * (kwargs["credit_or_debit"] != '') + \
		kwargs["credit_or_debit"] + "_"
	logging.info("Prefix is : {0}".format(prefix))
	#set file names
	files = {
		"test_poor" : prefix + "val_poor.csv",
		"train_poor" : prefix + "train_poor.csv"
	}
	#Write the poor CSVs
	poor_kwargs = {"header" : False, "index" : False, "index_label": False}
	kwargs["df_poor_test"].to_csv(files["test_poor"], **poor_kwargs)
	kwargs["df_poor_train"].to_csv(files["train_poor"], **poor_kwargs)
	#Return file names
	return files

def get_json_and_csv_files(**kwargs):
	"""This function generates CSV and JSON files"""
	prefix = kwargs["output_path"] + kwargs["bank_or_card"] + "_" + kwargs["credit_or_debit"] + "_"
	logging.info("Prefix is : {0}".format(prefix))
	#set file names
	files = {
		"map_file" : prefix + "map.json",
		"test_poor" : prefix + "test_poor.csv",
		"train_poor" : prefix + "train_poor.csv"
	}
	#Write the JSON file
	dict_2_json(kwargs["label_map"], files["map_file"])
	# Write the rich CSVs
	# rich_kwargs = {"index" : False, "sep" : "|"}
	# kwargs["df_test"].to_csv(files["test_rich"], **rich_kwargs)
	# kwargs["df_rich_train"].to_csv(files["train_rich"], **rich_kwargs)
	#Write the poor CSVs
	poor_kwargs = {"cols" : ["LABEL", "DESCRIPTION_UNMASKED"], "header": False,
		"index" : False, "index_label": False}
	kwargs["df_poor_test"].to_csv(files["test_poor"], **poor_kwargs)
	kwargs["df_poor_train"].to_csv(files["train_poor"], **poor_kwargs)
	#Return file names
	return files


def slice_into_dataframes(**kwargs):
	"""Slice into test and train dataframs, make a label map, and produce 
	CSV files."""
	# Create an output directory if it does not exist
	os.makedirs(kwargs["output_path"], exist_ok=True)
	# Load data frame and class names
	df, class_names = load(input_file=kwargs["input_file"], credit_or_debit=kwargs["credit_or_debit"])
	# Generate a mapping (class_name: label_number)
	label_map = get_label_map(class_names)
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

def copy_file(input_file, directory):
	"""This function moves uses Linux's 'cp' command to copy files on the local host"""
	logging.info("Copy the file {0} to directory: {1}".format(input_file, directory))
	local["cp"][input_file][directory]()

def fill_description_unmasked(row):
	"""Ensures that blank values for DESCRIPTION_UNMASKED are always filled."""
	if row["DESCRIPTION_UNMASKED"] == "":
		return row["DESCRIPTION"]
	return row["DESCRIPTION_UNMASKED"]

def unzip_and_merge(gz_file, bank_or_card):
	"""unzip an tar.gz file and merge all csv files into one,
	also returns a json label map"""
	directory = './merchant_' + bank_or_card + '_unzip/'
	os.makedirs(directory, exist_ok=True)
	label_maps = []
	local['tar']['xf'][gz_file]['-C'][directory]()
	merged = merge_csvs(directory)
	for i in os.listdir(directory):
		if i.endswith('.json'):
			label_maps.append(i)
	if len(label_maps) != 1:
		raise Exception('There does not exist a unique label map json file!.')
	return (merged, directory + label_maps[0])

def merge_csvs(directory):
	"merges all csvs immediately under the directory"
	dataframes = []
	cols = ["DESCRIPTION", "DESCRIPTION_UNMASKED", "MERCHANT_NAME"]
	for i in os.listdir(directory):
		if i.endswith('.csv'):
			df = load_piped_dataframe(directory + i, usecols=cols)
			dataframes.append(df)
	merged = pd.concat(dataframes, ignore_index=True)
	merged = check_empty_transaction(merged)
	return merged

def check_empty_transaction(df):
	"""Find transactions with empty description and return df with nonempty description"""
	empty_transaction = df[(df['DESCRIPTION_UNMASKED'] == '') &\
		(df['DESCRIPTION'] == '')]
	if len(empty_transaction) != 0:
		print("There are {0} empty transactions, \
			save to empty_transactions.csv"
			.format(len(empty_transaction)))
		empty_transaction.to_csv('empty_transactions.csv',
			sep='|', index=False)
	return df[(df['DESCRIPTION_UNMASKED'] != '') |\
			(df['DESCRIPTION'] != '')]

def seperate_debit_credit(csv_file, credit_or_debit, model_type):
	"""Load the CSV into a pandas data frame, return debit and credit df"""
	logging.info("Loading csv file")
	df = load_piped_dataframe(csv_file)
	df = check_empty_transaction(df)
	df['UNIQUE_TRANSACTION_ID'] = df.index
	df['LEDGER_ENTRY'] = df['LEDGER_ENTRY'].str.lower()
	if model_type == 'subtype':
		df["PROPOSED_SUBTYPE"] = df["PROPOSED_SUBTYPE"].str.strip()
		df['PROPOSED_SUBTYPE'] = df['PROPOSED_SUBTYPE'].apply(cap_first_letter)
	if model_type == 'category':
		df["PROPOSED_CATEGORY"] = df["PROPOSED_CATEGORY"].str.strip()
		df['PROPOSED_CATEGORY'] = df['PROPOSED_CATEGORY'].apply(cap_first_letter)
	grouped = df.groupby('LEDGER_ENTRY', as_index=False)
	groups = dict(list(grouped))
	return groups[credit_or_debit]

def reverse_map(label_map, key='label'):
	"""reverse {key : {category, label}} to {label: key} and
	{key: value} to {value: key} dictionary}"""
	get_key = lambda x: x[key] if isinstance(x, dict) else x
	reversed_label_map = dict(zip(map(get_key, label_map.values()),
		label_map.keys()))
	return reversed_label_map

if __name__ == "__main__":
	logging.error("Sorry, this module is a library of useful "
		"functions you can import into your code, you should not "
		"execute it from the command line.")
