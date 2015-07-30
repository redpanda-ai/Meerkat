#!/usr/local/bin/python3.3

"""This module takes processes a sample of transactions and outputs common
bigrams

Created on Jan 21, 2015
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# python3.3 -m meerkat.classification.generate_common_bigrams [transaction_sample] 
# python3.3 -m meerkat.classification.generate_common_bigrams data/card_sample.txt

# Required Columns: 
# DESCRIPTION_UNMASKED

#####################################################

import contextlib
import csv
import pandas as pd
import sys

from gensim.models import Phrases

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

	sample_included = (sample.endswith('.txt') or sample.endswith('.csv'))

	if not sample_included:
		safe_print("Erroneous arguments. Please see usage")
		sys.exit()

def add_local_params(params):
	"""Adds additional local params"""

	params["generate_common_bigrams"] = {
	}

	return params

def run_from_command_line(cla):
	"""Runs these commands if the module is invoked from the command line"""

	verify_arguments()
	params = {}
	params = add_local_params(params)
	first_chunk = True

	reader = pd.read_csv(cla[1], chunksize=5000, na_filter=False,\
	quoting=csv.QUOTE_NONE, encoding="utf-8", sep='|', error_bad_lines=False)

	# Process Transactions
	for chunk in reader:

		transactions = [row["DESCRIPTION_UNMASKED"].lower().split(" ") \
		for i, row in chunk.iterrows()]

		if first_chunk:
			bigrams = Phrases(transactions, max_vocab_size=5000)
		else:
			bigrams.add_vocab(transactions)

	bigrams.save("models/bigram_model")

if __name__ == "__main__":
	run_from_command_line(sys.argv)
