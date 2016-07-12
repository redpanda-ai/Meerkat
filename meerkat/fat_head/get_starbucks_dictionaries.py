import sys
import os
import csv
import logging
import pandas as pd
import json

def select_merchant_with_store(input_file, chunksize, merchant):
	dfs = []
	for chunk in pd.read_csv(input_file, chunksize=chunksize, error_bad_lines=False, warn_bad_lines=True,
		encoding='utf-8', quotechar='"', na_filter=False, sep=','):
		grouped = chunk.groupby('list_name', as_index=False)
		groups = dict(list(grouped))
		if merchant in groups.keys():
			dfs.append(groups[merchant])

	logging.info("start merging dataframes")
	merged = pd.concat(dfs, ignore_index=True)
	logging.info("finish merging dataframes")
	slender_df = merged[["store_number", "city", "state"]]
	first_dict, second_dict = {}, {}
	my_stores = slender_df.set_index("store_number").T.to_dict('list')
	for key in my_stores.keys():
		key_1, key_2 = key.split("-")
		first_dict[key_1] = my_stores[key]
		second_dict[key_2] = my_stores[key]

	first_json = json.dumps(first_dict, sort_keys=True, indent=4, separators=(',', ': '))
	second_json = json.dumps(second_dict, sort_keys=True, indent=4, separators=(',', ': '))
	print("First JSON:\n{0}".format(first_json))
	print("Second JSON:\n{0}".format(second_json))

select_merchant_with_store("All_Merchants.csv", 1000, "Starbucks")
