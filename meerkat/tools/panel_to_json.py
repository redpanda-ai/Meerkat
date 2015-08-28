"""This module samples panel files to produce JSON."""

#################### USAGE ##########################

# python3.3 panel_to_json.py [panel_file] [number_of_transactions]
# python3.3 panel_to_json.py card_sample_3_years_large.txt 1500

#####################################################

import sys
import re
import pandas as pd
import numpy as np
import json

def dict_2_json(obj, filename):
	"""Saves a dict as a json file"""
	with open(filename, 'w') as output_file:
		json.dump(obj, output_file, indent=4)

def string_cleanse(original_string):
	"""Strips out characters that might confuse ElasticSearch."""
	bad_characters = [r"'", r"\*<"]
	bad_character_regex = "|".join(bad_characters)
	cleanse_pattern = re.compile(bad_character_regex)
	with_spaces = re.sub(cleanse_pattern, " ", original_string)
	return ' '.join(with_spaces.split())

def well_named_function():
	"""This function brings it all together."""
	query = {
		"cobrand_id" : 1234,
		"user_id" : 2345,
		"container" : "card",
		"transaction_list" : []
	}

	data_frame = pd.read_csv(sys.argv[1], na_filter=False, encoding="utf-8",
		 sep='|', error_bad_lines=False)

	rows = np.random.choice(data_frame.index.values, int(sys.argv[2]),
		replace=False)
	sampled_df = data_frame.ix[rows]

	for index, row in sampled_df.iterrows():
		trans = {
			"amount" : 100.00,
			"date" : row["TRANSACTION_DATE"],
			"ledger_entry" : "debit",
			"transaction_id" : int(index),
			"description" : row["DESCRIPTION_UNMASKED"]
		}
		query["transaction_list"].append(trans)

	dict_2_json(query, "web_service_input.json")

if __name__ == "main":
	well_named_function()
