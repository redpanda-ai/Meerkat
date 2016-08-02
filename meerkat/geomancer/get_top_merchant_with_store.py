import sys
import os
import csv
import logging
import pandas as pd

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
	merged.to_csv(merchant + '_with_store.csv', sep=',', index=False)

select_merchant_with_store("All_Merchants.csv", 1000, "Starbucks")
