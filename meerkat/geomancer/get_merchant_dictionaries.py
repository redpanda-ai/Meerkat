import argparse
import csv
import inspect
import json
import os
import pandas as pd
import json
import queue
import sys
import logging
import yaml

from functools import reduce
from timeit import default_timer as timer
from .tools import get_etags, get_s3_file, remove_special_chars, get_grouped_dataframes
from .get_top_merchant_data import extract_clean_files
from .geomancer_module import GeomancerModule
from meerkat.various_tools import load_params

logging.config.dictConfig(yaml.load(open('meerkat/geomancer/logging.yaml', 'r')))
logger = logging.getLogger('get_merchant_dictionaries')

def preprocess_dataframe(df):
	"""Fix up some of the data in our dataframe."""
	capitalize_word = lambda x: x.upper()
	df["state"] = df["state"].apply(capitalize_word)
	df["city"] = df["city"].apply(capitalize_word)
	df["city"] = df["city"].apply(expand_abbreviations)

def merge(a, b, path=None):
	"""Useful function to merge complex dictionaries, courtesy of:
	https://stackoverflow.com/questions/7204805/dictionaries-of-dictionaries-merge
	I added logic for handling merge conflicts so that new values overwrite old values
	but throw warnings.
	"""
	if path is None:
		path = []
	for key in b:
		if key in a:
			if isinstance(a[key], dict) and isinstance(b[key], dict):
				merge(a[key], b[key], path + [str(key)])
			elif a[key] == b[key]:
				pass
			else:
				#Merge conflict, in our case old beats new
				logger.warning("Conflict at %s" % ".".join(path + [str(key)]))
				logger.warning("A key {0}".format(a[key]))
				logger.warning("B key {0}".format(b[key]))
				logger.warning("Conflict in dicts, keeping older value.")
		else:
			a[key] = b[key]
	return a

def expand_abbreviations(city):
	"""Turns abbreviations into their expanded form."""
	maps = {
		"E. ": "EAST ", "W. ": "WEST ", "N. ": "NORTH ", "S. ": "SOUTH ",
		"ST. ": "SAINT ", "ST ": "SAINT ", "FT. ": "FORT ", "FT ": "FORT "
	}
	for abbr in maps:
		if city.startswith(abbr):
			city = city.replace(abbr, maps[abbr])
			break
	return city

class Worker(GeomancerModule):
	"""Contains methods and data pertaining to the creation and retrieval of merchant dictionaries"""
	name = "merchant_dictionaries"

	def __init__(self, common_config, config):
		"""Constructor"""
		super(Worker, self).__init__(common_config, config)
		module_path = inspect.getmodule(inspect.stack()[1][0]).__file__
		self.filepath = module_path[:module_path.rfind("/") + 1] + "merchants"

	def get_store_dictionaries(self, df):
		"""Writes out two store dictionaries"""
		logger.debug("Generating store dictionaries.")
		#Use only the "store_number", "city", and "state" columns
		slender_df = df[["store_number", "city", "state"]]
		store_dict_1, store_dict_2 = {}, {}
		my_stores = slender_df.set_index("store_number").T.to_dict('list')
		#my_stores = {str(k):str(v) for k,v in my_stores.items()}
		#Split the store_id dicts
		for key in my_stores.keys():
			key = str(key)
			#If each key cannot be split by a dash, return the full my_stores_dictionary
			if key.count("-") == 0:
				self.dump_pretty_json_to_file(my_stores, "store_id_1.json")
				self.dump_pretty_json_to_file(my_stores, "store_id_2.json")
				return my_stores, my_stores
			#Otherwise, build a split dictionary
			key_1, key_2 = key.split("-")
			store_dict_1[key_1] = my_stores[key]
			store_dict_2[key_2] = my_stores[key]
		#Dump the store_id dictionaries
		self.dump_pretty_json_to_file(store_dict_1, "store_id_1.json")
		self.dump_pretty_json_to_file(store_dict_2, "store_id_2.json")
		#Return the dictionaries
		return store_dict_1, store_dict_2

	def get_unique_city_dictionaries(self, df):
		"""Constructs a dictionary using unique city names as keys."""
		logger.debug("Generating unique_city dictionaries for {0}".format(self.config["merchant"]))
		# Create the unique_city_state dictionary
		grouped_city = self.geo_df.groupby('city', as_index=True)
		groups_city = dict(list(grouped_city))
		unique_city_state = {}
		for city, group in groups_city.items():
			states = group.state.unique()
			if len(states) == 1:
				unique_city_state[city.upper()] = states[0].upper()
		# Write the unique_city_state dictionary to json file
		merchant = self.config["merchant"]
		self.dump_pretty_json_to_file(unique_city_state, "unique_city_state.json")
		# Create the unique_city list
		unique_city = list(unique_city_state.keys())
		# Write the unique_city list to json file
		self.dump_pretty_json_to_file(unique_city, "unique_city.json")
		return unique_city_state, unique_city

	def get_geo_dictionary(self, df):
		"""Generates three merchant dictionaries and writes them as JSON files"""
		merchant = self.config["merchant"]
		logger.debug("Generating geo dictionaries for '{0}'".format(self.config["merchant"]))
		#Create a geo-dictionary, using only "state", "city", and "zip_code"
		geo_df = df[["state", "city", "zip_code"]]
		grouped = geo_df.groupby(['state', 'city'], as_index=True)
		geo_dict = {}
		for name, group in grouped:
			state, city = name
			state, city = state.upper(), city.upper()
			if state not in geo_dict:
				geo_dict[state] = {}
			if city not in geo_dict[state]:
				geo_dict[state][city] = []
			for item in group["zip_code"]:
				item = str(item)
				item = item[:5]
				if item not in geo_dict[state][city]:
					geo_dict[state][city].append(item)
		#Write the geo-dictionary
		self.dump_pretty_json_to_file(geo_dict, "geo.json")
		#Return the dataframe
		return geo_df

	def main_process(self):
		"""This is where it all happens."""
		bucket = self.common_config["bucket"]
		prefix = self.config["prefix"]
		file_name = self.config["filename"]
		save_path = self.config["savepath"]
		os.makedirs(save_path, exist_ok=True)

		etags, etags_file = get_etags(save_path)
		logger.info("Synch-ing with S3")
		needs_to_be_downloaded = get_s3_file(bucket=bucket, prefix=prefix, file_name=file_name,
			save_path=save_path, etags=etags, etags_file=etags_file)
		logger.info("Synch-ed")

		if needs_to_be_downloaded:
			extract_clean_files(save_path, file_name, "csv")

		target_merchants = self.common_config["target_merchant_list"]
		merchants_map = load_params("meerkat/geomancer/merchant_name_map.json")
		merchants_reverse_map = {}
		for merchant_in_cnn in target_merchants:
			for merchant in merchants_map[merchant_in_cnn]:
				merchants_reverse_map[merchant] = merchant_in_cnn
		merchants_in_agg_data = merchants_reverse_map.keys()
		logger.info("Merchants in agg data: {0}".format(merchants_in_agg_data))
		logger.info("Merchant reverse map: {0}".format(merchants_reverse_map))

		merchant_dataframes = {}
		for file_name in os.listdir(save_path):
			if file_name.endswith(".csv"):
				csv_file = save_path + file_name
				logger.info("csv file at: " + csv_file)
				csv_kwargs = { "chunksize": 1000, "error_bad_lines": False, "warn_bad_lines": True,
					"encoding": "utf-8-sig", "quotechar" : '"', "na_filter" : False, "sep": "," }
				merchant_dataframes_per_file = get_grouped_dataframes(csv_file,
					"list_name", merchants_in_agg_data, **csv_kwargs)
				merchant_dataframes = merge(merchant_dataframes, merchant_dataframes_per_file)

		merchants_found_in_agg_data = sorted(list(merchant_dataframes.keys()))
		logger.info("Merchants found in agg data: {0}".format(merchants_found_in_agg_data))
		merchants = set()
		formatted_merchant_dataframes = {}
		for merchant in merchants_found_in_agg_data:
			merchant_in_cnn = merchants_reverse_map[merchant]
			if merchant_in_cnn in merchants:
				formatted_merchant_dataframes[merchant_in_cnn] = formatted_merchant_dataframes[merchant_in_cnn].\
					append(merchant_dataframes[merchant], ignore_index=True)
			else:
				merchants.add(merchant_in_cnn)
				formatted_merchant_dataframes[merchant_in_cnn] = merchant_dataframes[merchant]
		merchants = list(merchants)

		logger.warning("Found {0} target merchants: {1}".format(len(merchants), merchants))
		missed_list = list(set(self.common_config["target_merchant_list"]) - set(merchants))
		logger.warning("Miss {0} target merchants: {1}".format(len(missed_list), missed_list))

		self.common_config["target_merchant_list"] = merchants

		for merchant in merchants:
			self.config["merchant"] = remove_special_chars(merchant)
			df = formatted_merchant_dataframes[merchant]
			preprocess_dataframe(df)
			logger.info("***** Processing {0:>29} ********".format(merchant))
			self.setup_directories()
			store_dict_1, store_dict_2 = self.get_store_dictionaries(df)
			self.geo_df = self.get_geo_dictionary(df)
			unique_city_state, unique_city = self.get_unique_city_dictionaries(df)
			# Let's also get the question bank
		return self.common_config

	def setup_directories(self):
		"""This creates the directories on the local file system, if needed."""
		if not self.config["dry_run"]:
			output_directory = self.filepath + "/" + self.config["merchant"]
			logger.debug("Confirming output directory at {0}".format(output_directory))
			os.makedirs(output_directory, exist_ok=True)
		else:
			logger.info("No need for output directory for {0}".format(self.config["merchant"]))

	def dump_pretty_json_to_file(self, new_object, filename):
		"""Dumps a pretty-printed JSON object to the file provided."""
		src_object = {}
		dst_object = {}
		full_path = self.filepath + "/" + self.config["merchant"] + "/" + filename
		try:
			with open(full_path, "r") as infile:
				src_object = json.load(infile)
		except IOError as e:
			logger.debug("No pre-existing object, which is fine.")
		#Merge original and new new_object
		if isinstance(new_object, dict):
			dst_object = reduce(merge, [src_object, new_object])
			dst_object = {str(k): v for k, v in dst_object.items()}
		elif isinstance(new_object, list):
			dst_object = list(set().union(new_object, src_object))
		else:
			logger.critical("It's neither a list nor a dictionary, aborting.")
			sys.exit()
		#Dump, if necessary
		log_write = "'" + self.config["merchant"] + "/" + filename + "'"
		if not self.config["dry_run"]:
			logger.info("Writing {0}".format(log_write))
			with open(full_path, "w") as outfile:
				json.dump(dst_object, outfile, sort_keys=True, indent=4, separators=(',', ': '))
		else:
			logger.info("Not Writing {0}".format(log_write))


if __name__ == "__main__":
	logger.critical("You cannot run this from the command line, aborting.")

