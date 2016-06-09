"""This module preprocess data and return a 2-column csv file for the Crepe algorithm"""

import sys
import os
import json
import pandas as pd
import numpy as np

from meerkat.various_tools import load_piped_dataframe

#################### USAGE ##########################

# python3 -m meerkat.tools.merchant_process [container]
# python3 -m meerkat.tools.merchant_process bank

#####################################################

def dict_2_json(obj, filename):
	"""Saves a dict as a json file"""
	with open(filename, 'w') as reader:
		json.dump(obj, reader, indent=4)

def run_from_command_line():
	"""Runs these commands if the module is invoked from the command line"""

	directory = "./" + sys.argv[1] + "_CNN_merchant_training_samples"
	samples = []

	# Get Samples
	for i in os.listdir(directory):
		if i.endswith(".csv"):
			if i == "1.csv":
				null_class = i
			else:
				samples.append(i)

	# Create New Label Map
	new_label_nums = list(range(2, (len(samples) + 2)))
	label_num_map = dict(zip(samples, new_label_nums))
	label_num_map[null_class] = "1"
	dataframes = []
	output_map = {}

	# Process Data
	for sample in samples:
		df = load_piped_dataframe(directory + "/" + sample)
		df['LABEL_NUM'] = str(label_num_map[sample])
		dataframes.append(df)
		output_map[label_num_map[sample]] = df["MERCHANT_NAME"][0]

	# Add Null Class
	null_df = load_piped_dataframe(directory + "/1.csv")
	null_df['LABEL_NUM'] = "1"
	dataframes.append(null_df)
	output_map[1] = ""

	# Merge Dataframes
	merged = pd.concat(dataframes, ignore_index=True)

	# Drop Empty Rows and Unnecessary Columns
	merged['DESCRIPTION_UNMASKED'] = merged.apply(lambda x: x["DESCRIPTION_UNMASKED"].strip(), axis=1)
	merged = merged[merged["DESCRIPTION_UNMASKED"] != ""]
	final = merged[['LABEL_NUM', 'DESCRIPTION_UNMASKED']]

	# Make Test and Train
	msk = np.random.rand(len(final)) < 0.90
	train = final[msk]
	test = final[~msk]
	train_full = merged[msk]
	test_full = merged[~msk]
	train_full["dataset"] = "train"
	test_full["dataset"] = "test"
	full_dataset = pd.concat([test_full, train_full], ignore_index=True)

	# Save
	dict_2_json(output_map, "new_rlm_" + sys.argv[1] + ".json")
	train.to_csv("train_" + sys.argv[1] + ".csv", header=False, index=False, index_label=False)
	test.to_csv("test_" + sys.argv[1] + ".csv", header=False, index=False, index_label=False)
	full_dataset.to_csv(sys.argv[1] + "_full.csv", index=False, sep="|")

if __name__ == "__main__":
	run_from_command_line()
