import argparse
import csv
import inspect
import sys
import os
import logging
import pandas as pd
import json

from functools import reduce

def parse_arguments(args):
	"""Parses arguments"""
	module_path = inspect.getmodule(inspect.stack()[1][0]).__file__
	default_path = module_path[:module_path.rfind("/") + 1] + "dictionaries"
	parser = argparse.ArgumentParser()
	parser.add_argument("--merchant", default="Starbucks")
	parser.add_argument("--filepath", default=default_path)

	dumping_parser = parser.add_mutually_exclusive_group(required=False)
	dumping_parser.add_argument('--dumping', dest='dumping', action='store_true')
	dumping_parser.add_argument('--no-dumping', dest='dumping', action='store_false')
	parser.set_defaults(dumping=False)
	return parser.parse_args(args)

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
				#Merge conflict, let the new guy 
				logging.warning("Conflict at %s" % ".".join(path + [str(key)]))
				logging.warning("A key {0}".format(a[key]))
				logging.warning("B key {0}".format(b[key]))
				logging.warning("Conflict in dicts, resolving with newer value.")
		else:
			a[key] = b[key]
	return a

def dump_pretty_json_to_file(new_object, filename):
	"""Dumps a pretty-printed JSON object to the file provided."""
	src_object = {}
	dst_object = {}
	full_path = ARGS.filepath + "/" + ARGS.merchant + "/" + filename
	try:
		with open(full_path, "r") as infile:
			src_object = json.load(infile)
	except IOError as e:
		logging.debug("No pre-existing object, which is fine.")
	#Merge original and new new_object
	if isinstance(new_object, dict):
		dst_object = reduce(merge, [src_object, new_object])
		dst_object = {str(k): v for k, v in dst_object.items()}
	elif isinstance(new_object, list):
		dst_object = list(set().union(new_object, src_object))
	else:
		logging.critical("It's neither a list nor a dictionary, aborting.")
		sys.exit()
	#Dump, if necessary
	log_write = "'" + ARGS.merchant + "/" + filename + "'"
	if ARGS.dumping:
		logging.info("Writing {0}".format(log_write))
		with open(full_path, "w") as outfile:
			json.dump(dst_object, outfile, sort_keys=True, indent=4, separators=(',', ': '))
	else:
		logging.info("Not Writing {0}".format(log_write))

def expand_abbreviations(city):
	"""Turns abbreviations into their expanded form."""
	maps = {
		"E. ": "EAST ",
		"W. ": "WEST ",
		"N. ": "NORTH ",
		"S. ": "SOUTH ",

		"ST. ": "SAINT ",
		"ST ": "SAINT ",
		"FT. ": "FORT ",
		"FT ": "FORT "
	}
	for abbr in maps:
		if city.startswith(abbr):
			city = city.replace(abbr, maps[abbr])
			break
	return city

def get_merchant_dataframes(input_file, chunksize):
	"""Generate a dataframe which is a subset of the input_file grouped by merchant."""
	logging.info("Constructing dataframe from file.")
	#Here are the target merchants
	target_merchants = [ "Ace Hardware", "Walmart", "Walgreens", "Target",
		"Subway", "Starbucks", "McDonald's", "Costco Wholesale Corp.", "Burger King",
		"Bed Bath and Beyond",
		"Aeropostale", "Albertsons", "American Eagle Outfitters", "Applebee's", "Arby's",
		"AutoZone", "Bahama Breeze", "Barnes & Noble", "Baskin-Robbins", "Bealls",
		"Eddie V's", "Fedex", "Five Guys", "Food 4 Less", "Francesca's", "Fred Meyer",
		"Gymboree", "H&M", "Home Depot", "IHOP", "In-N-Out Burger", "J. C. Penney",
		"KFC", "Kmart", "Kohl's", "LongHorn Steakhouse", "Lowe's", "Macy's", "Nordstrom"
	]
	#create a list of dataframe groups, filtered by merchant name
	dict_of_df_lists = {}
	dfs = []
	chunk_num = 0
	#logging.info("Filtering by the following merchant: {0}".format(merchant))
	for chunk in pd.read_csv(input_file, chunksize=chunksize, error_bad_lines=False,
		warn_bad_lines=True, encoding='utf-8', quotechar='"', na_filter=False, sep=','):
		chunk_num += 1
		if chunk_num % 10 == 0:
			logging.info("Processing chunk {0:07d}, {1:>6} target merchants found.".format(chunk_num,
				len(dict_of_df_lists.keys())))
		grouped = chunk.groupby('list_name', as_index=False)
		groups = dict(list(grouped))
		#logging.info("Group Keys: {0}".format(groups.keys()))
		my_keys = groups.keys()
		for key in my_keys:
			if key in target_merchants:
				if key not in dict_of_df_lists:
					logging.info("***** Discovered {0:>30} ********".format(key))
					dict_of_df_lists[key] = []
				dict_of_df_lists[key].append(groups[key])

	#Show what you found and did not find
	merchants_found = dict_of_df_lists.keys()
	found_list = list(merchants_found)
	missing_list = list(set(target_merchants) - set(found_list))
	for item in found_list:
		logging.info("Found {0:>49}".format(item))
	for item in missing_list:
		logging.warning("Not Found {0:>42}".format(item))
	#Merge them together
	for key in merchants_found:
		dict_of_df_lists[key] = pd.concat(dict_of_df_lists[key], ignore_index=True)
		#Do some pre-processing
		logging.info("Preprocessing dataframe for {0:>27}".format(key))
		preprocess_dataframe(dict_of_df_lists[key])

	return dict_of_df_lists

def get_store_dictionaries(df):
	"""Writes out two store dictionaries"""
	logging.debug("Generating store dictionaries.")
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
			dump_pretty_json_to_file(my_stores, "store_id_1.json")
			dump_pretty_json_to_file(my_stores, "store_id_2.json")
			return my_stores, my_stores
		#Otherwise, build a split dictionary
		key_1, key_2 = key.split("-")
		store_dict_1[key_1] = my_stores[key]
		store_dict_2[key_2] = my_stores[key]
	#Dump the store_id dictionaries
	dump_pretty_json_to_file(store_dict_1, "store_id_1.json")
	dump_pretty_json_to_file(store_dict_2, "store_id_2.json")
	#Return the dictionaries
	return store_dict_1, store_dict_2

def preprocess_dataframe(df):
	"""Fix up some of the data in our dataframe."""
	capitalize_word = lambda x: x.upper()
	df["state"] = df["state"].apply(capitalize_word)
	df["city"] = df["city"].apply(capitalize_word)
	df["city"] = df["city"].apply(expand_abbreviations)

def get_unique_city_dictionaries(df):
	"""Constructs a dictionary using unique city names as keys."""
	logging.debug("Generating unique_city dictionaries for {0}".format(ARGS.merchant))
	# Create the unique_city_state dictionary
	grouped_city = geo_df.groupby('city', as_index=True)
	groups_city = dict(list(grouped_city))
	unique_city_state = {}
	for city, group in groups_city.items():
		states = group.state.unique()
		if len(states) == 1:
			unique_city_state[city.upper()] = states[0].upper()
	# Write the unique_city_state dictionary to json file
	merchant = ARGS.merchant
	dump_pretty_json_to_file(unique_city_state, "unique_city_state.json")
	# Create the unique_city list
	unique_city = list(unique_city_state.keys())
	# Write the unique_city list to json file
	dump_pretty_json_to_file(unique_city, "unique_city.json")
	return unique_city_state, unique_city

def get_geo_dictionary(df):
	"""Generates three merchant dictionaries and writes them as JSON files"""
	merchant = ARGS.merchant
	logging.debug("Generating geo dictionaries for '{0}'".format(ARGS.merchant))
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
	dump_pretty_json_to_file(geo_dict, "geo.json")
	#Return the dataframe
	return geo_df

def setup_directories():
	"""This creates the directories on the local file system, if needed."""
	if ARGS.dumping:
		output_directory = ARGS.filepath + "/" + ARGS.merchant
		logging.debug("Confirming output directory at {0}".format(output_directory))
		os.makedirs(output_directory, exist_ok=True)
	else:
		logging.info("No need for output directory for {0}".format(ARGS.merchant))

if __name__ == "__main__":
	log_format = "%(asctime)s %(levelname)s: %(message)s"
	logging.basicConfig(format=log_format, level=logging.INFO)
	ARGS = parse_arguments(sys.argv[1:])
	merchant_dataframes = get_merchant_dataframes("All_Merchants.csv", 1000)
	merchants = sorted(list(merchant_dataframes.keys()))
	for merchant in merchants:
		ARGS.merchant = merchant
		df = merchant_dataframes[merchant]
		logging.info("***** Processing {0:>29} ********".format(merchant))
		setup_directories()
		store_dict_1, store_dict_2 = get_store_dictionaries(df)
		geo_df = get_geo_dictionary(df)
		unique_city_state, unique_city = get_unique_city_dictionaries(df)

