#!/usr/local/bin/python3

"""Train a CNN using tensorFlow

Created on Mar 14, 2016
@author: Matthew Sevrens
@author: Tina Wu
"""

#################### USAGE #######################

# python3 -m meerkat.classification.tf_CNN

##################################################

import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import json
from .verify_data import load_json
from .tools import fill_description_unmasked

input_file = "Card_complete_data_subtype_original_updated_credit.csv"
df = pd.read_csv(input_file, quoting=csv.QUOTE_NONE, na_filter=False,
		encoding="utf-8", sep='|', error_bad_lines=False)

df['LEDGER_ENTRY'] = df['LEDGER_ENTRY'].str.lower()
grouped = df.groupby('LEDGER_ENTRY', as_index=False)
groups = dict(list(grouped))
df = groups["credit"]
# Clean the "DESCRIPTION_UNMASKED" values within the dataframe
df["DESCRIPTION_UNMASKED"] = df.apply(fill_description_unmasked, axis=1)
print(len(df))

label_map = "card_credit_subtype_label_map.json"
label_map = load_json(label_map)

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
ALPHA_DICT = {a : i for i, a in enumerate(ALPHABET)}

def string_to_tensor(str, l):
	"""Convert transaction to tensor format"""

	s = str.lower()[0:l]
	t = np.zeros((len(ALPHABET), l), dtype=np.float32)
	for i, c in reversed(list(enumerate(s))):
		if c in ALPHABET:
			t[ALPHA_DICT[c]][len(s) - i - 1] = 1
	return t

def build_cnn():
	"""Build CNN"""


	graph = tf.Graph()
	num_labels = 10
	batch_size = 128
	doc_length = 123
	alphabet_length = len(ALPHABET)

	# Create Graph
	with graph.as_default():

		x = tf.placeholder(tf.float32, shape=[batch_size, 1, alphabet_length, doc_length])
		y = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
		w = tf.Variable(tf.random_normal([1, alphabet_length, doc_length, 256], name="W"))
		tf.nn.conv2d(x, w, [1,1,1,1], padding="SAME")

def run_session(graph):
	"""Run Session"""

	# Train Network
	epochs = 5000
	eras = 10

	with tf.Session(graph=graph) as session:

		tf.initialize_all_variables().run()
		num_eras = epochs * eras

		for step in range(num_eras):

			if (step % epochs == 0):

				print("Save details")

if __name__ == "__main__":
	build_cnn()
