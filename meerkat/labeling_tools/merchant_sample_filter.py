#!/usr/local/bin/python3.3

"""This module takes a sample of a possible transactions for a single
merchant and allows a reviewer to filter out non matching transactions

Created on Jan5, 2015
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# Note: In Progress
# python3.3 -m meerkat.labeling_tools.merchant_sample_filter [merchant_sample] 
# python3.3 -m meerkat.labeling_tools.merchant_sample_filter data/misc/Merchant\ Samples/16K_Target_Card.txt

# Required Columns: 
# DESCRIPTION_UNMASKED
# UNIQUE_MEM_ID
# MERCHANT_NAME
# GOOD_DESCRIPTION
# UNIQUE_TRANSACTION_ID

#####################################################

import contextlib
import csv
import sys

import pandas as pd
import numpy as np

from meerkat.various_tools import safe_print, safe_input

class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostderr():
    save_stderr = sys.stderr
    sys.stderr = DummyFile()
    yield
    sys.stderr = save_stderr

def verify_arguments():
	"""Verify Usage"""

	sufficient_arguments = (len(sys.argv) == 2)

	if not sufficient_arguments:
		safe_print("Insufficient arguments. Please see usage")
		sys.exit()

	sample = sys.argv[1]

	sample_included = sample.endswith('.txt')

	if not sample_included:
		safe_print("Erroneous arguments. Please see usage")
		sys.exit()

def add_local_params(params):
	"""Adds additional local params"""

	params["merchant_sample_filter"] = {
	}

	return params

def run_from_command_line(cla):
	"""Runs these commands if the module is invoked from the command line"""

	verify_arguments()
	params = {}
	params = add_local_params(params)
	df = pd.read_csv(cla[1], na_filter=False, quoting=csv.QUOTE_NONE, encoding="utf-8", sep='|', error_bad_lines=False)
	sub_df = df[["DESCRIPTION_UNMASKED", "MERCHANT_NAME"]]
	sLen = len(sub_df['DESCRIPTION_UNMASKED'])
	labeler = safe_input("What is the Yodlee email of the current labeler?\n")

	# Add a new column if first time labeling this data set
	if labeler not in df.columns:
		sub_df[labeler] = pd.Series(([""] * sLen))
	else:
		sub_df[labeler] = df[labeler]

	safe_print(sub_df)

	# Step 5: Loop through each row (until completion or save out) and prompt for 1: Is this Merchant, 0: Is not this merchant, 2: Skip - Not Sure 
	# Step 6: On key to save to file, map decision column with username as header back to dataframe and save out file
	
if __name__ == "__main__":
	run_from_command_line(sys.argv)
