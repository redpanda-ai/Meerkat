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

from collections import defaultdict
from boto.s3.connection import Key, Location
import pandas as pd
import numpy as np

from meerkat.various_tools import safely_remove_file, clean_bad_escapes, load_params, get_s3_connection

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

def run_from_command_line(cla):
	"""Runs these commands if the module is invoked from the command line"""

	# Connect to S3
	with nostdout():
		conn = get_s3_connection()
		
	bucket = conn.get_bucket("yodleeprivate", Location.USWest2)
	columns = ["DESCRIPTION_UNMASKED", "DECRIPTION", "GOOD_DESCRIPTION", "TRANSACTION_DATE", "UNIQUE_TRANSACTION_ID", "AMOUNT", "UNIQUE_MEM_ID", "TYPE"]
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

	# Sample Files
	for i, item in enumerate(files):
		file_name = "data/output/" + os.path.basename(item.key)
		try:
			item.get_contents_to_filename(file_name)
			reader = pd.read_csv(file_name, na_filter=False, chunksize=1000000, dtype=dtypes, compression="gzip", quoting=csv.QUOTE_NONE, encoding="utf-8", sep='|', error_bad_lines=False)
			
			for df in reader:

				df['MERCHANT_NAME'] = df.apply(map_labels, axis=1)
				grouped = df.groupby('MERCHANT_NAME', as_index=False)
				groups = dict(list(grouped))

				# Apply Reservoir Sampling Over Each Merchant
				for merchant, merchant_df in groups.items():
					
					n = 1000000 if merchant == "" else SAMPLE_SIZE
					merchant_file_name = "data/output/s3_sample/" + num_map[merchant]
					
					# Create Dataframe if file doesn't exist
					try:
						output_df = pd.read_csv(merchant_file_name, na_filter=False, dtype=dtypes, quoting=csv.QUOTE_NONE, encoding="utf-8", sep='|', error_bad_lines=False)
					except:
						output_df = pd.DataFrame(columns=columns)

					# Apply Reservoir Sampling
					for row in merchant_df.iterrows():
						merchant_count[merchant] += 1
						count = merchant_count[merchant]
						row_to_add = [row[c] for c in columns]
						if count < n:
							output_df = output_df.append(row_to_add, ignore_index=True)
						elif count >= n and random.random() < n / count:
							i = random.randint(0, n)
							output_df.loc[i] = row_to_add

					# Save Output		
					output_df.to_csv(merchant_file_name, columns=columns, sep="|", mode="w", encoding="utf-8", index=False, index_label=False)
			
			del files[i]
			safely_remove_file(file_name)

		except:
			safely_remove_file(file_name)
			continue

if __name__ == "__main__":
	run_from_command_line(sys.argv)