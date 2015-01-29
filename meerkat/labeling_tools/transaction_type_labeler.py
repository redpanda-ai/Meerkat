#!/usr/local/bin/python3.3

"""This module takes a sample of a transactions and allows multiple
labelers to assign a type and subtype to each transaction

Created on Jan 5, 2015
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# Note: In Progress
# python3.3 -m meerkat.labeling_tools.transaction_type_labeler [merchant_sample] 
# python3.3 -m meerkat.labeling_tools.transaction_type_labeler data/misc/transaction_type_GT_Card.txt

# Required Columns: 
# DESCRIPTION_UNMASKED
# UNIQUE_MEM_ID
# UNIQUE_TRANSACTION_ID
# AMOUNT
# TRANSACTION_BASE_TYPE

#####################################################

import contextlib
import csv
import sys
import os

import pandas as pd
from boto.s3.connection import Key, Location

from meerkat.various_tools import safe_print, safe_input, load_params
from meerkat.producer import connect_to_S3

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

def identify_container(filename):
	"""Determines whether transactions are bank or card"""

	if "bank" in filename.lower():
		return "bank"
	elif "card" in filename.lower():
		return "card"
	else:
		print('Please designate whether this is bank or card in params["container"]')
		sys.exit()


def move_to_S3(params, bucket, s3_path, filepath, labeler):
	"""Moves a file to S3"""

	filename = os.path.basename(filepath)
	s3_path = s3_path + params["container"] + "/"

	key = Key(bucket)
	key.key = s3_path + labeler + "_" + filename
	bytes_written = key.set_contents_from_filename(filepath, encrypt_key=True, replace=True)
	safe_print("File written to: " + key.key)
	safe_print("{0} bytes written".format(bytes_written))
	#safely_remove_file(filepath)

def add_local_params(params):
	"""Adds additional local params"""

	params["merchant_sample_filter"] = {}
	params["container"] = identify_container(sys.argv[1].lower())

	return params

def run_from_command_line(cla):
	"""Runs these commands if the module is invoked from the command line"""

	verify_arguments()
	params = load_params("config/labeling_prototype.json")
	params = add_local_params(params)
	conn = connect_to_S3()
	bucket = conn.get_bucket(params["S3"]["bucket_name"], Location.USWest2)
	df = pd.read_csv(cla[1], na_filter=False, quoting=csv.QUOTE_NONE, encoding="utf-8", sep='|', error_bad_lines=False)
	sLen = df.shape[0]
	labeler = safe_input("What is the Yodlee email of the current labeler?\n")
	tt_col = labeler + "_TT"
	st_col = labeler + "_ST"

	# Add new columns if first time labeling this data set
	if (tt_col) not in df.columns:
		df[tt_col] = pd.Series(([""] * sLen))

	if (st_col) not in df.columns:
		df[st_col] = pd.Series(([""] * sLen))

	# Capture Decisions
	save_and_exit = False
	choices = [c["name"] for c in params["labels"]]
	sub_choices = [s for s in params["labels"] if "sub_labels" in s]
	sub_dict = {}
	skip_save = ["", "s"]
	options = skip_save + [str(o) for o in list(range(0, len(choices)))]

	# Create Loopup for Sub types
	for sub in sub_choices:
		sub_dict[sub["name"]] = sub["sub_labels"]

	while "" in df[tt_col].tolist():

		for index, row in df.iterrows():

			# Skip rows that already have decisions
			if row[tt_col] in choices:
				if row[tt_col] in sub_dict:
					if row[st_col] in sub_dict[row[tt_col]]:
						continue
				else: 
					continue
			

			# Collect labeler choice
			choice = None
			sub_choice = None
			safe_print(("_" * 75) + "\n")

			# Show Progress
			percent_complete = ((sLen - df[tt_col].str.contains(r'^$').sum()) / sLen) * 100
			safe_print("{0:.2f}%".format(percent_complete) + " complete with labeling\n")

			# Show transaction details
			for c in params["display_columns"]:
				safe_print("{}: {}".format(c, row[c]))

			# Prompt with top level question
			safe_print("\nWhich of the following transaction types best describes the preceding transaction?\n")
			
			# Prompt with choices
			for i, item in enumerate(choices):
				safe_print("{:7s} {}".format("[" + str(i) + "]", item))
			
			safe_print("\n[enter] Skip")
			safe_print("{:7s} Save and Exit".format("[s]"))
			
			while choice not in options:
				choice = safe_input()
				if choice not in options:
					safe_print("Please select one of the options listed above")

			# Prompt for subtype if neccesary
			choice_name = choices[int(choice)] if choice not in ["", "s"] else choice

			if choice_name in sub_dict:

				sub_options = skip_save + [str(o) for o in list(range(0, len(sub_dict[choice_name])))]

				safe_print("\nWhich of the following subtypes best describes the preceding transaction?\n")

				for i, item in enumerate(sub_dict[choice_name]):
					safe_print("{:7s} {}".format("[" + str(i) + "]", item))

				safe_print("\n[enter] Skip")
				safe_print("{:7s} Save and Exit".format("[s]"))

				while sub_choice not in sub_options:
					sub_choice = safe_input()
					if sub_choice not in sub_options:
						safe_print("Please select one of the options listed above")

			if choice_name == "s" or sub_choice == "s":
				save_and_exit = True
				break

			# Enter choices into decision matrix
			df.loc[index, tt_col] = "" if choice_name == "" else choice_name

			if sub_choice != None:
				df.loc[index, st_col] = "" if sub_choice == "" else sub_dict[choice_name][int(sub_choice)]

		# Break if User exits
		if save_and_exit:
			df.to_csv(sys.argv[1], sep="|", mode="w", encoding="utf-8", index=False, index_label=False)
			move_to_S3(params, bucket, "development/labeled/", sys.argv[1], labeler)
			break
	
if __name__ == "__main__":
	run_from_command_line(sys.argv)
