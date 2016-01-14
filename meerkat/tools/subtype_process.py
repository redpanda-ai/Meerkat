#!/usr/local/bin/python3.3

import os
import sys
import json
import csv

import numpy as np
import pandas as pd

#################### USAGE ##########################

# python3 process.py [file_name] [ledger_entry]
# python3 process.py Bank_complete_data_subtype_original.csv credit

#####################################################

def dict_2_json(obj, filename):
	"""Saves a dict as a json file"""
	with open(filename, 'w') as fp:
		json.dump(obj, fp, indent=4)

def load_label_map(filename):
	"""Load a permanent label map"""
	input_file = open(filename, encoding='utf-8')
	label_map = json.loads(input_file.read())
	input_file.close()
	return label_map

training_data = []

# Load Data
df = pd.read_csv(sys.argv[1], quoting=csv.QUOTE_NONE, na_filter=False,
	encoding="utf-8", sep='|', error_bad_lines=False, low_memory=False)
df['UNIQUE_TRANSACTION_ID'] = df.index
df['LEDGER_ENTRY'] = df['LEDGER_ENTRY'].str.lower()
grouped = df.groupby('LEDGER_ENTRY', as_index=False)
groups = dict(list(grouped))
df = groups[sys.argv[2]]
df["PROPOSED_SUBTYPE"] = df["PROPOSED_SUBTYPE"].str.strip().str.lower()
class_names = df["PROPOSED_SUBTYPE"].value_counts().index.tolist()

# Create a label map
label_numbers = list(range(1, (len(class_names) + 1)))
label_map = dict(zip(sorted(class_names), label_numbers))

# Map Numbers
a = lambda x: label_map[x["PROPOSED_SUBTYPE"]]
df['LABEL'] = df.apply(a, axis=1)

# Replace Description_Unmasked
def b(x):
	if x["DESCRIPTION_UNMASKED"] == "":
		return x["DESCRIPTION"]
	else:
		return x["DESCRIPTION_UNMASKED"]
df["DESCRIPTION_UNMASKED"] = df.apply(b, axis=1)

# Make Test and Train
training_data_df = df[['LABEL', 'DESCRIPTION_UNMASKED']]
msk = np.random.rand(len(training_data_df)) < 0.90
train = training_data_df[msk]
train_full = df[msk]
test = training_data_df[~msk]
test_full = df[~msk]

# Save
label_map = dict(zip(label_map.values(), label_map.keys()))
dict_2_json(label_map, 'data/preprocessed/' + sys.argv[2] + "_subtype_label_map.json")
train.to_csv('data/preprocessed/' + sys.argv[2] + "_train_subtype.csv",
	 cols=["LABEL", "DESCRIPTION_UNMASKED"], header=False, index=False,
	 index_label=False)
test.to_csv('data/preprocessed/' + sys.argv[2] + "_test_subtype.csv",
	 cols=["LABEL", "DESCRIPTION_UNMASKED"], header=False, index=False,
	 index_label=False)

test_full.to_csv('data/preprocessed/' + sys.argv[2] + "_test_subtype_full.csv",
	 index=False, sep='|')
train_full.to_csv('data/preprocessed/' + sys.argv[2] + "_train_subtype_full.csv",
	 index=False, sep='|')
