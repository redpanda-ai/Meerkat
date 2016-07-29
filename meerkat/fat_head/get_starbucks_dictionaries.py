import argparse
import sys
import inspect
import os
import csv
import logging
import pandas as pd
import json

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

def dump_pretty_json_to_file(dictionary, filename):
	"""Dumps a pretty-printed JSON object to the file provided."""
	if ARGS.dumping:
		full_path = ARGS.filepath + "/" + ARGS.merchant + "/" + filename
		logging.info("Dumping {0}".format(full_path))
		with open(full_path, "w") as outfile:
			json.dump(dictionary, outfile, sort_keys=True, indent=4, separators=(',', ': '))
	else:
		logging.info("Not Dumping {0}".format(filename))

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
	merchant = ARGS.merchant
	dict_of_df_lists = {}
	dfs = []
	chunk_num = 0
	logging.info("Filtering by the following merchant: {0}".format(merchant))
	for chunk in pd.read_csv(input_file, chunksize=chunksize, error_bad_lines=False,
		warn_bad_lines=True, encoding='utf-8', quotechar='"', na_filter=False, sep=','):
		chunk_num += 1
		if chunk_num % 10 == 0:
			logging.info("Processing chunk {0:>4}, {1:>4} target merchants found.".format(chunk_num,
				len(dict_of_df_lists.keys())))
		grouped = chunk.groupby('list_name', as_index=False)
		groups = dict(list(grouped))
		#logging.info("Group Keys: {0}".format(groups.keys()))
		my_keys = groups.keys()
		for key in my_keys:
			if key in target_merchants:
				if key not in dict_of_df_lists:
					logging.info("Adding {0}".format(key))
					dict_of_df_lists[key] = []
				dict_of_df_lists[key].append(groups[key])

	#Show what you found and did not find
	merchants_found = dict_of_df_lists.keys()
	found_list = list(merchants_found)
	missing_list = list(set(target_merchants) - set(found_list))
	logging.info("Found List: {0}".format(found_list))
	logging.info("Missing List: {0}".format(missing_list))

	#Merge them together
	for key in merchants_found:
		dict_of_df_lists[key] = pd.concat(dict_of_df_lists[key], ignore_index=True)
	df = dict_of_df_lists[merchant]
	#df = pd.concat(dfs, ignore_index=True)
	#Do some pre-processing
	logging.info("Preprocessing dataframe.")
	preprocess_dataframe(df)
	#Return the dataframe
	return dict_of_df_lists

def get_store_dictionaries(df):
	"""Writes out two store dictionaries"""
	logging.info("Generating store dictionaries.")
	#Use only the "store_number", "city", and "state" columns
	slender_df = df[["store_number", "city", "state"]]
	store_dict_1, store_dict_2 = {}, {}
	my_stores = slender_df.set_index("store_number").T.to_dict('list')
	#Split the store_id dicts
	for key in my_stores.keys():
		key = str(key)
		#If each key cannot be split by a dash, return the full my_stores_dictionary
		if key.count("-") == 0:
			return my_stores, my_stores
		#Otherwise, build a split dictionary
		key_1, key_2 = key.split("-")
		store_dict_1[key_1] = my_stores[key]
		store_dict_2[key_2] = my_stores[key]
	#Dump the store_id dictionaries
	merchant = ARGS.merchant
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
	logging.info("Generating unique_city dictionaries for {0}".format(ARGS.merchant))
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
	logging.info("Generating geo dictionaries for '{0}'".format(ARGS.merchant))
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
	if ARGS.dumping:
		output_directory = ARGS.filepath + "/" + ARGS.merchant
		logging.info("Confirming output directory at {0}".format(output_directory))
		os.makedirs(output_directory, exist_ok=True)
	else:
		logging.info("No need for output directory for {0}".format(ARGS.merchant))

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	ARGS = parse_arguments(sys.argv[1:])
	merchant_dataframes = get_merchant_dataframes("All_Merchants.csv", 1000)
	merchants = sorted(list(merchant_dataframes.keys()))
	logging.info("Merchants {0}".format(merchants))
	for merchant in merchants:
		ARGS.merchant = merchant
		df = merchant_dataframes[merchant]
		logging.info("Processing '{0}'".format(merchant))
		setup_directories()
		store_dict_1, store_dict_2 = get_store_dictionaries(df)
		geo_df = get_geo_dictionary(df)
		unique_city_state, unique_city = get_unique_city_dictionaries(df)

