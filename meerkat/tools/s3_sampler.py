#!/usr/local/bin/python3.3

"""This module monitors S3 and our file processing farm of EC2 instances.
If it notices new input files to process, it does so.  So long as there
are available 'slots'.  The remaining input files are kept in a stack.

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

from meerkat.various_tools import safely_remove_file, clean_bad_escapes
from meerkat.file_producer import get_s3_connection

#################### USAGE ##########################

# Note: In Progress
# python3.3 -m meerkat.tools.s3_sampler

# yodleeprivate/ctprocessed/gpanel/card/
# yodleeprivate/panels/meerkat_split/bank/

#####################################################

class DummyFile(object):
	def write(self, x): 
		pass

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
	bank_regex = re.compile("panels/meerkat_split/bank/")
	card_regex = re.compile("ctprocessed/gpanel/card/")
	bank_columns = ["DESCRIPTION_UNMASKED", "GOOD_DESCRIPTION", \
	"TRANSACTION_DATE", "UNIQUE_TRANSACTION_ID"]
	card_columns = ["DESCRIPTION_UNMASKED", "GOOD_DESCRIPTION", \
	"TRANSACTION_DATE", "UNIQUE_TRANSACTION_ID"]
	dtypes = {"DESCRIPTION_UNMASKED":"object", "GOOD_DESCRIPTION":"object", \
	"TRANSACTION_DATE":"object", "UNIQUE_TRANSACTION_ID":"object"}
	bank_files, card_files = [], []
	first_chunk = True

	# Get a list of files
	for panel in bucket:
		if bank_regex.search(panel.key) and os.path.basename(panel.key) != "":
			bank_files.append(panel)
		if card_regex.search(panel.key) and os.path.basename(panel.key) != "":
			card_files.append(panel)

	print("Number of bank files " + str(len(bank_files)))

	# Sample Card Files
	for i, item in enumerate(bank_files):
		file_name = "/mnt/ephemeral/sampling/bank/" + os.path.basename(item.key)
		try:
			item.get_contents_to_filename(file_name)

			dataframe = pd.read_csv(file_name, na_filter=False, chunksize=100000, \
			dtype=dtypes, compression="gzip", quoting=csv.QUOTE_NONE, encoding="utf-8", \
			sep='|', error_bad_lines=False)
			
			for df in dataframe:

				num_to_sample = math.ceil(len(df.index) * 0.0075)
			
				print("Sampled " + str(num_to_sample) + " transactions from file")
				rows = np.random.choice(df.index.values, num_to_sample)
				sampled_df = df.ix[rows]

				if first_chunk:
					sampled_df[bank_columns].to_csv("/mnt/ephemeral/sampling/bank/bank_sample.txt", \
					columns=bank_columns, sep="|", mode="a", encoding="utf-8", \
					index=False, index_label=False)
					first_chunk = False
				else:
					sampled_df[bank_columns].to_csv("/mnt/ephemeral/sampling/bank/bank_sample.txt", \
					header=False, columns=bank_columns, sep="|", mode="a", encoding="utf-8", \
					index=False, index_label=False)
			
			del card_files[i]
			safely_remove_file(file_name)

		except:
			safely_remove_file(file_name)
			continue

if __name__ == "__main__":
	run_from_command_line(sys.argv)
