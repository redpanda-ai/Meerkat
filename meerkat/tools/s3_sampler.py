#!/usr/local/bin/python3.3

"""This module samples a collection of panel files and creates a
training set for the CNN merchant classifier

Created on Apr 13, 2015
@author: Matthew Sevrens
"""

import sys
import math
import json
import csv
import re
import os
import contextlib
import random
import psutil

from collections import defaultdict
from boto.s3.connection import Key, Location
import pandas as pd
import numpy as np

from meerkat.various_tools import (safely_remove_file, clean_bad_escapes,
	load_params, get_s3_connection)

SAMPLE_SIZE = 10000

#################### USAGE ##########################

# Note: In Progress
# python3.3 -m meerkat.tools.s3_sampler [container]
# python3.3 -m meerkat.tools.s3_sampler bank

# yodleeprivate/ctprocessed/gpanel/card/
# yodleeprivate/panels/meerkat_split/bank/

#####################################################

class DummyFile(object):
	def write(self, x): pass

@contextlib.contextmanager
def nostdout():
	save_stdout = sys.stdout
	save_stderr = sys.stderr
	sys.stdout = DummyFile()
	sys.stderr = DummyFile()
	yield
	sys.stderr = save_stderr
	sys.stdout = save_stdout

def save_df(df, file_name, columns):
	"""Save a dataframe"""
	df.to_csv(file_name, columns=columns, sep="|", mode="w", quoting=csv.QUOTE_NONE, encoding="utf-8", index=False, index_label=False)

def load_df(file_name, dtypes):
	"""Load a dataframe"""
	return pd.read_csv(file_name, na_filter=False, dtype=dtypes, quoting=csv.QUOTE_NONE, encoding="utf-8", sep='|', error_bad_lines=False)

def save_cache(cache, columns):
	"""Save files in cache"""
	for file_name, df in cache.items():
		save_df(df, file_name, columns)

def run_from_command_line(cla):
	"""Runs these commands if the module is invoked from the command line"""

	# Connect to S3
	with nostdout():
		conn = get_s3_connection()
		
	bucket = conn.get_bucket("yodleeprivate", Location.USWest2)
	columns = ["DESCRIPTION_UNMASKED", "DESCRIPTION", "GOOD_DESCRIPTION", "TRANSACTION_DATE", "UNIQUE_TRANSACTION_ID", "AMOUNT", "UNIQUE_MEM_ID", "TYPE"]
	dtypes = {x: "object" for x in columns}
	columns.append("MERCHANT_NAME")
	files = []

	# Create Output Folder if it doesn't exist
	os.makedirs("data/output/s3_sample", exist_ok=True)

	if sys.argv[1] == "bank":
		regex = re.compile("panels/meerkat_split/bank/")
		label_map = load_params("meerkat/classification/label_maps/expanded_permanent_bank_label_map.json")
		reverse_label_map = load_params("meerkat/classification/label_maps/expanded_reverse_bank_label_map.json")
	elif sys.argv[1] == "card":
		regex = re.compile("ctprocessed/gpanel/card/")
		label_map = load_params("meerkat/classification/label_maps/expanded_permanent_card_label_map.json")
		reverse_label_map = load_params("meerkat/classification/label_maps/expanded_reverse_card_label_map.json")
	else: 
		print("Please select bank or card for container")
		sys.exit()

	# Get a list of files
	for panel in bucket:
		if regex.search(panel.key) and os.path.basename(panel.key) != "":
			files.append(panel)

	# Build Label Map
	ct_to_cnn_map = {}
	for key, value in label_map.items():
		if str(value) in reverse_label_map:
			ct_to_cnn_map[key] = reverse_label_map[str(value)]

	print("Number of " + sys.argv[1] + " files " + str(len(files)))
	num_map = dict(zip(reverse_label_map.values(), reverse_label_map.keys()))
	map_labels = lambda x: ct_to_cnn_map.get(x["GOOD_DESCRIPTION"].lower(), "")
	merchant_count = defaultdict(lambda: 0)
	df_cache = {}
	cache_full = False

	# Sample Files
	for i, item in enumerate(files):
		file_name = "data/output/" + os.path.basename(item.key)
		
		try:
			item.get_contents_to_filename(file_name)
			reader = pd.read_csv(file_name, na_filter=False, chunksize=500000, dtype=dtypes, compression="gzip", quoting=csv.QUOTE_NONE, encoding="utf-8", sep='|', error_bad_lines=False)
			first_null_chunk = True

			for df in reader:

				print("chunk")

				# Replace CT labels with well formatted labels
				df['MERCHANT_NAME'] = df.apply(map_labels, axis=1)
				grouped = df.groupby('MERCHANT_NAME', as_index=False)
				groups = dict(list(grouped))

				# Apply Reservoir Sampling Over Each Merchant
				for merchant, merchant_df in groups.items():

					merchant_df = merchant_df[columns]
					output_df = None
					
					n = SAMPLE_SIZE
					merchant_file_name = "data/output/s3_sample/" + num_map[merchant] + ".csv"

					# Sample Null Class Differently for performance reasons
					if merchant == "":
						num_to_sample = math.ceil(len(merchant_df.index) * 0.0075)
						rows = np.random.choice(merchant_df.index.values, num_to_sample)
						sampled_df = merchant_df.ix[rows]
						if first_null_chunk:
							sampled_df.to_csv(merchant_file_name, columns=columns, sep="|", mode="a", encoding="utf-8", index=False, index_label=False)
							first_null_chunk = False
						else: 
							sampled_df.to_csv(merchant_file_name, header=False, columns=columns, sep="|", mode="a", encoding="utf-8", index=False, index_label=False)
						continue

					# Create Dataframe if file doesn't exist
					if merchant_file_name in df_cache:
						output_df = df_cache[merchant_file_name]
					elif merchant_count[merchant] == 0:
						output_df = pd.DataFrame(columns=columns)
					elif merchant_count[merchant] < n:
						output_df = load_df(merchant_file_name, dtypes)		

					# Fill Reservoirs
					if merchant_count[merchant] < n:
						o_len = len(output_df)
						m_len = len(merchant_df)
						if o_len + m_len <= n:
							output_df = output_df.append(merchant_df, ignore_index=False)
							merchant_count[merchant] += m_len
							save_df(output_df, merchant_file_name, columns)
							continue
						else:
							r = n - o_len
							output_df = output_df.append(merchant_df.iloc[0:r], ignore_index=False)
							merchant_count[merchant] += r
							merchant_df = merchant_df.iloc[r+1:m_len-1]
							save_df(output_df, merchant_file_name, columns)
				
					rows_to_add = []

					# Select Rows to Replace
					for k in merchant_df.index:
						merchant_count[merchant] += 1
						rand = random.random()
						if rand < n / merchant_count[merchant]:
							rows_to_add.append(k)

					if len(rows_to_add) > 0:

						# Load df if not loaded yet
						if output_df is None:
							output_df = load_df(merchant_file_name, dtypes)
						
						# Replace Rows
						output_df.iloc[np.random.choice(range(n), len(rows_to_add))] = merchant_df.loc[rows_to_add].values

						# Save or Cache Output
						if cache_full and merchant_file_name not in df_cache:
							save_df(output_df, merchant_file_name, columns)
						else:
							df_cache[merchant_file_name] = output_df
							mem = psutil.virtual_memory().percent
							if mem > 90:
								cache_full = True

			del files[i]
			safely_remove_file(file_name)

		except:
			safely_remove_file(file_name)
			continue

	save_cache(df_cache, columns)

if __name__ == "__main__":
	run_from_command_line(sys.argv)
