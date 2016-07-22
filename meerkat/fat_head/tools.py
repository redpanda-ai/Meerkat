
import argparse
import csv
import json
import logging
import sys
import pandas as pd

def parse_arguments(args):
	parser = argparse.ArgumentParser()
	parser.add_argument("input_file")
	parser.add_argument("--subset", default="")
	return parser.parse_args(args)

def deduplicate_csv(input_file, subset):
	"""This function de-deduplicates transactions in a csv file"""
	df = pd.read_csv(input_file, error_bad_lines=False,
		encoding='utf-8', na_filter=False, sep=',')

	if subset == "":
		unique_df = df.drop_duplicates(keep="first")
	else:
		unique_df = df.drop_duplicates(subset=subset, keep="first")

	logging.info("reduced {0} duplicate transactions".format(len(df) - len(unique_df)))
	output_file = 'deduplicated_' + input_file
	unique_df.to_csv(output_file, sep=',', index=False, quoting=csv.QUOTE_ALL)
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

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	logging.critical("Do not run this module from the command line.")
