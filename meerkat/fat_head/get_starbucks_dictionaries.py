import sys
import re
import os
import csv
import logging
import pandas as pd
import json

def generate_merchant_dictionaries(input_file, chunksize, merchant):
	"""Generates three merchant dictionaries and writes them as JSON files"""
	#create a list of dataframe groups, filtered by merchant name
	logging.warning("Processing CSV file.")
	dfs = []
	for chunk in pd.read_csv(input_file, chunksize=chunksize, error_bad_lines=False, warn_bad_lines=True,
		encoding='utf-8', quotechar='"', na_filter=False, sep=','):
		grouped = chunk.groupby('list_name', as_index=False)
		groups = dict(list(grouped))
		if merchant in groups.keys():
			dfs.append(groups[merchant])

	logging.warning("start merging dataframes")
	#Merge them together
	merged = pd.concat(dfs, ignore_index=True)
	logging.warning("finish merging dataframes")

	#Use only the "store_number", "city", and "state" columns
	slender_df = merged[["store_number", "city", "state"]]
	first_dict, second_dict = {}, {}
	my_stores = slender_df.set_index("store_number").T.to_dict('list')

	#Split the store_id dicts
	for key in my_stores.keys():
		key_1, key_2 = key.split("-")
		first_dict[key_1] = my_stores[key]
		second_dict[key_2] = my_stores[key]

	#Dump the store_id dictionaries
	logging.warning("Dumping store_id_*.json files.")
	with open(merchant + "_store_id_1.json", "w") as outfile:
		json.dump(first_dict, outfile, sort_keys=True, indent=4, separators=(',', ': '))
	with open(merchant + "_store_id_2.json", "w") as outfile:
		json.dump(second_dict, outfile, sort_keys=True, indent=4, separators=(',', ': '))

	#Create a geo-dictionary, using only "state", "city", and "zip_code"
	geo_df = merged[["state", "city", "zip_code"]]
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
			item = item[:5]
			if item not in geo_dict[state][city]:
				geo_dict[state][city].append(item)

	#Write the geo-dictionary
	logging.warning("Dumping _geo.json files.")
	with open(merchant + "_geo.json", "w") as outfile:
		json.dump(geo_dict, outfile, sort_keys=True, indent=4, separators=(',', ': '))
	first_json = json.dumps(first_dict, sort_keys=True, indent=4, separators=(',', ': '))

	# Create the unique_city_state dictionary
	grouped_city = geo_df.groupby('city', as_index=True)
	groups_city = dict(list(grouped_city))
	unique_city_state = {}
	for city, group in groups_city.items():
		states = group.state.unique()
		if len(states) == 1:
			unique_city_state[city.upper()] = states[0].upper()

	#Write the unique_city_state dictionary
	logging.warning("Dumping unique_city_state.json files.")
	with open(merchant + "_unique_city_state.json", "w") as outfile:
		json.dump(unique_city_state, outfile, sort_keys=True, indent=4, separators=(',', ': '))

generate_merchant_dictionaries("All_Merchants.csv", 1000, "Starbucks")
