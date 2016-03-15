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

import sys
import csv
import json
import math
import random

from .verify_data import load_json
from .tools import fill_description_unmasked, reverse_map

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
ALPHA_DICT = {a : i for i, a in enumerate(ALPHABET)}
NUM_LABELS = 10
BATCH_SIZE = 128
DOC_LENGTH = 123
ALPHABET_LENGTH = len(ALPHABET)

def load_data():
	"""Load data and label map"""

	label_map = "card_credit_subtype_label_map.json"
	label_map = load_json(label_map)
	reversed_map = reverse_map(label_map)
	a = lambda x: reversed_map.get(str(x["PROPOSED_SUBTYPE"]), "")

	input_file = "Card_complete_data_subtype_original_updated_credit.csv"
	df = pd.read_csv(input_file, quoting=csv.QUOTE_NONE, na_filter=False, encoding="utf-8", sep='|', error_bad_lines=False)

	df['LEDGER_ENTRY'] = df['LEDGER_ENTRY'].str.lower()
	grouped = df.groupby('LEDGER_ENTRY', as_index=False)
	groups = dict(list(grouped))
	df = groups["credit"]
	df["DESCRIPTION_UNMASKED"] = df.apply(fill_description_unmasked, axis=1)
	df = df.reindex(np.random.permutation(df.index))
	df["LABEL_NUM"] = df.apply(a, axis=1)
	df = df[df["LABEL_NUM"] != ""]

	batched = np.array_split(df, math.ceil(df.shape[0] / 128))

	return label_map, batched

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

	# Create Graph
	with graph.as_default():

		x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 1, ALPHABET_LENGTH, DOC_LENGTH])
		y = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
		w = tf.Variable(tf.random_normal([1, ALPHABET_LENGTH, DOC_LENGTH, 256], name="W"))
		tf.nn.conv2d(x, w, [1,1,1,1], padding="SAME")

	# Run Graph
	run_session(graph)

def run_session(graph):
	"""Run Session"""

	# Train Network
	label_map, batched = load_data()
	epochs = 5000
	eras = 10

	with tf.Session(graph=graph) as session:

		tf.initialize_all_variables().run()
		num_eras = epochs * eras

		for step in range(num_eras):

			batch = random.choice(batched)
			labels = np.array(batched[0]["LABEL_NUM"].astype(int))
			labels = (np.arange(NUM_LABELS) == labels[:,None]).astype(np.float32)
			print(labels)

			if (step % epochs == 0):

				print("Save details")

			sys.exit()

if __name__ == "__main__":
	build_cnn()
