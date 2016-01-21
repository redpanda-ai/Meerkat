#!/usr/local/bin/python3.3

"""This script loads raw bank subtype data and partitions it into training
and test data. It also produces a index-label mapping json file.

@author: Matthew Sevrens
@author: Oscar Pan
"""

#################### USAGE ##########################
"""
python3 process.py [file_name] [ledger_entry]
python3 -m meerkat.tools.subtype_process \
data/input/Bank_complete_data_subtype_original.csv credit bank
"""
#####################################################

import os
import sys
import json
import csv
import string

import numpy as np
import pandas as pd


def dict_2_json(obj, filename):
	"""Saves a dict as a json file"""
	with open(filename, 'w') as fp:
		json.dump(obj, fp, indent=4)

def cap_first_letter(label):
	"""Make sure the first letter of each word is capitalized"""
	temp = label.split()
	for i in range(len(temp)):
		if temp[i].lower() in ['by', 'with', 'or', 'at', 'in']:
			temp[i] = temp[i].lower()
		else:
			temp[i] = temp[i][0].upper() + temp[i][1:]
	return ' '.join(word for word in temp)

training_data = []

# Load Data
df = pd.read_csv(sys.argv[1], quoting=csv.QUOTE_NONE, na_filter=False,
	encoding="utf-8", sep='|', error_bad_lines=False, low_memory=False)
df['UNIQUE_TRANSACTION_ID'] = df.index
df['LEDGER_ENTRY'] = df['LEDGER_ENTRY'].str.lower()
grouped = df.groupby('LEDGER_ENTRY', as_index=False)
groups = dict(list(grouped))
df = groups[sys.argv[2]]
df["PROPOSED_SUBTYPE"] = df["PROPOSED_SUBTYPE"].str.strip()
df['PROPOSED_SUBTYPE'] = df['PROPOSED_SUBTYPE'].apply(cap_first_letter)
class_names = df["PROPOSED_SUBTYPE"].value_counts().index.tolist()

# Create a label map
label_numbers = list(range(1, (len(class_names) + 1)))
label_map = dict(zip(sorted(class_names), label_numbers))

# Map Numbers
a = lambda x: label_map[x["PROPOSED_SUBTYPE"]]
df['LABEL'] = df.apply(a, axis=1)

# Replace Description_Unmasked
def fill_description(row):
	if row["DESCRIPTION_UNMASKED"] == "":
		return row["DESCRIPTION"]
	else:
		return row["DESCRIPTION_UNMASKED"]
df["DESCRIPTION_UNMASKED"] = df.apply(fill_description, axis=1)

# Make Test and Train
training_data_df = df[['LABEL', 'DESCRIPTION_UNMASKED']]
msk = np.random.rand(len(training_data_df)) < 0.90
train = training_data_df[msk]
train_full = df[msk]
test = training_data_df[~msk]
test_full = df[~msk]

# Save
# Check if a dir exsits, if not create one
path = 'data/preprocessedd/'
os.makedirs(path, exist_ok=True)

label_map = dict(zip(label_map.values(), label_map.keys()))
dict_2_json(label_map, path + sys.argv[3] + '_' + sys.argv[2]
	 + "_subtype_label_map.json")
train.to_csv(path + sys.argv[3] + '_' + sys.argv[2] + "_train_subtype.csv",
	 cols=["LABEL", "DESCRIPTION_UNMASKED"], header=False, index=False,
	 index_label=False)
test.to_csv(path + sys.argv[3] + '_' + sys.argv[2] + "_test_subtype.csv",
	 cols=["LABEL", "DESCRIPTION_UNMASKED"], header=False, index=False,
	 index_label=False)

test_full.to_csv(path + sys.argv[3] + '_' + sys.argv[2] + "_test_subtype_full.csv",
	 index=False, sep='|')
train_full.to_csv(path + sys.argv[3] + '_' + sys.argv[2] + "_train_subtype_full.csv",
	 index=False, sep='|')
