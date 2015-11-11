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

from boto.s3.connection import Key, Location
import pandas as pd
import numpy as np

from meerkat.various_tools import safely_remove_file, clean_bad_escapes, load_params, get_s3_connection

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
	first_chunk = True

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
	output_file = "data/output/" + sys.argv[1] + "_sample.txt"
	map_labels = lambda x: ct_to_cnn_map.get(x["GOOD_DESCRIPTION"], "")

	# Sample Files
	for i, item in enumerate(files):
		file_name = "data/output/" + os.path.basename(item.key)
		try:
			item.get_contents_to_filename(file_name)
			reader = pd.read_csv(file_name, na_filter=False, chunksize=100000, dtype=dtypes, compression="gzip", quoting=csv.QUOTE_NONE, encoding="utf-8", sep='|', error_bad_lines=False)
			
			for df in reader:

				df['MERCHANT_NAME'] = df.apply(map_labels, axis=1)
				grouped = df.groupby('MERCHANT_NAME', as_index=False)
				groups = dict(list(grouped))

				for merchant, merchant_df in groups.items():
					print(merchant + ": " + str(df.shape[1]))
					sys.exit()

				num_to_sample = math.ceil(len(df.index) * 0.0075)
			
				print("Sampled " + str(num_to_sample) + " transactions from file")
				rows = np.random.choice(df.index.values, num_to_sample)
				sampled_df = df.ix[rows]

				if first_chunk:
					sampled_df[columns].to_csv(output_file, columns=columns, sep="|", mode="a", encoding="utf-8", index=False, index_label=False)
					first_chunk = False
				else:
					sampled_df[columns].to_csv(output_file, header=False, columns=columns, sep="|", mode="a", encoding="utf-8", index=False, index_label=False)
			
			del files[i]
			safely_remove_file(file_name)

		except:
			safely_remove_file(file_name)
			continue

if __name__ == "__main__":
	run_from_command_line(sys.argv)