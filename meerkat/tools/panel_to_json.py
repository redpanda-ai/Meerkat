"""This module samples panel files to produce JSON for use
with the web service."""

#################### USAGE ##########################

# python3 -m meerkat.tools.panel_to_json [panel_file] [number_of_transactions]
# python3 -m meerkat.tools.panel_to_json card_sample_3_years_large.txt 1500

#####################################################

import sys
import re
import numpy as np
import json

from meerkat.various_tools import load_piped_dataframe

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

def sample_panel_to_json():
	"""This function brings it all together."""
	query = {
		"cobrand_id" : 1234,
		"user_id" : 2345,
		"container" : "card",
		"transaction_list" : []
	}

	data_frame = load_piped_dataframe(sys.argv[1])

	rows = np.random.choice(data_frame.index.values, int(sys.argv[2]), replace=False)
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

if __name__ == '__main__':
	sample_panel_to_json()
