import re
import argparse
import csv
import json
import logging
import sys
import shutil
import pandas as pd

def remove_special_chars(input_string):
	"""Remove special characters in the input strint"""
	return re.sub(r"[ |\-|'|.|&]", r'', input_string)

def parse_arguments(args):
	parser = argparse.ArgumentParser()
	parser.add_argument("input_file")
	parser.add_argument("--subset", default="")
	return parser.parse_args(args)

def deduplicate_csv(input_file, subset, inplace, **csv_kwargs):
	"""This function de-deduplicates transactions in a csv file"""
	read = csv_kwargs["read"]
	to = csv_kwargs["to"]
	df = pd.read_csv(input_file, **read)
	original_len = len(df)

	if subset == "":
		unique_df = df.drop_duplicates(keep="first", inplace=inplace)
	else:
		unique_df = df.drop_duplicates(subset=subset, keep="first", inplace=inplace)

	if inplace:
		logging.info("reduced {0} duplicate transactions".format(original_len - len(df)))
		df.to_csv(input_file, **to)
		logging.info("csv files with unique {0} transactions saved to: {1}".format(len(df), input_file))
	else:
		logging.info("reduced {0} duplicate transactions".format(len(df) - len(unique_df)))
		last_slosh = input_file.rfind("/")
		output_file = input_file[: last_slosh + 1] + 'deduplicated_' + input_file[last_slosh + 1 :]
		unique_df.to_csv(output_file, **to)
		logging.info("csv files with unique {0} transactions saved to: {1}".format(len(unique_df), output_file))

def get_geo_dictionary(input_file):
	"""This function takes a csv file containing city, state, and zip and creates
	a dictionary."""
	my_dict = {}
	with open(input_file) as infile:
		for line in infile:
			parts = line.split("\t")
			city = parts[2].upper()
			state = parts[4].upper()
			zipcode = parts[1]
			if state not in my_dict:
				my_dict[state] = {}
			if city not in my_dict[state]:
				my_dict[state][city] = [zipcode]
			else:
				my_dict[state][city].append(zipcode)

	my_json = json.dumps(my_dict, sort_keys=True, indent=4, separators=(',', ': '))
	return my_json

def copy_file(input_file, directory):
	"""This function moves uses Linux's 'cp' command to copy files on the local host"""
	logging.info("Copy the file {0} to directory: {1}".format(input_file, directory))
	shutil.copy(input_file, directory)

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	logging.critical("Do not run this module from the command line.")
