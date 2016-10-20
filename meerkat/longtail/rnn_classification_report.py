#/usr/local/bin/python3.3
# pylint: disable=used-before-assignment
# pylint: disable=too-many-locals

"""This module loads and evaluates a trained RNN (biLSTM) on a provided test set.
It produces various stats and a confusion matrix for analysis

@author: Oscar Pan
"""

############################# USAGE #############################

# python3 -m meerkat.longtail.rnn_classification_report \
# <path_to_data> <path_to_cktp_file> <path_to_w2i_file> \
# --config <optional_path_to_config_file>

#################################################################

import logging
import argparse
import sys
import os
import time
import pandas as pd
import numpy as np
from itertools import groupby
from collections import defaultdict

from meerkat.classification.load_model import get_tf_rnn_by_path
from meerkat.various_tools import load_piped_dataframe
from meerkat.classification.classification_report import (get_classification_report,
	count_transactions)
from meerkat.longtail.bilstm_tagger import validate_config, tokenize
from meerkat.classification.tools import reverse_map

def parse_arguments(args):
	"""Create parser"""
	parser = argparse.ArgumentParser(description="Test a RNN and return performance statistics")
	# Required arguments
	parser.add_argument("data", help="Path to the test data")
	parser.add_argument("model", help="Path to the model under test")
	parser.add_argument("w2i", help="Path to the model's word to indice json file")
	# Optional arguments
	parser.add_argument("--config", help="Path to the config json file",
		default="./meerkat/longtail/bilstm_config.json")
	return parser.parse_args(args)

def beautify(item, config):
	"""make item easier to read"""

	item.pop("ground_truth")
	output = [config["tag_map"][str(i)] for i in np.argmax(item["Predicted"], 1)]

	tran = tokenize(item["DESCRIPTION"])
	tagged = list(zip(tran, output))
	grouped, dict_output = [], defaultdict(list)

	# Group Sequential Tokens
	for tag, group in groupby(tagged, lambda x: x[1]):
		merged = " ".join([x[0] for x in group])
		grouped.append((tag, merged))

	# Create Dict
	for x in grouped:
		dict_output[x[0]].append(x[1])

	# Add To Output
	for tag, tokens in dict(dict_output).items():
		item["predicted_" + tag] = ", ".join(tokens)

	del item["Predicted"]

	return item

def get_write_func(file_path, config):
	"""return a data writing function"""
	file_exist = False
	def write_func(data):
		"""save data to csv file"""
		if len(data) > 0:
			nonlocal file_exist
			mode = "a" if file_exist else "w"
			header = False if file_exist else True
			data = [beautify(item, config) for item in data]
			logging.info("Saving transactions to {0}".format(file_path))
			df = pd.DataFrame(data)
			df.to_csv(file_path, sep="|", mode=mode, index=False, header=header)
			file_exist = True
	return write_func

def evaluate_model(args=None):
	"""evaluates model accuracy and reports various statistics"""

	os.makedirs("./data/RNN_stats/", exist_ok=True)

	if args is None:
		args = parse_arguments(sys.argv[1:])

	config = validate_config(args.config)
	num_labels = len(config["tag_map"])
	con_matrix = [[0] * num_labels for i in range(num_labels)]
	reader = load_piped_dataframe(args.data, chunksize=1000)
	model = get_tf_rnn_by_path(args.model, args.w2i)
	total_trans = count_transactions(args.data)
	processed = 0.0
	save_mislabeled = get_write_func("data/RNN_stats/mislabeled.csv", config)
	save_correct = get_write_func("data/RNN_stats/correct.csv", config)
	elapsed_time = 0

	for chunk in reader:
		processed += len(chunk)
		mislabeled = []
		correct = []
		logging.info("Processing {0:3.2f}% of the data...".format(100*processed/total_trans))
		chunk = chunk.to_dict("record")

		start = time.time() 
		chunk = model(chunk, name_only=False, tags=True, doc_key="DESCRIPTION")
		end = time.time()
		elapsed_time += end - start

		for item in chunk:
			columns = [i for i in np.argmax(item["Predicted"], 1)]
			rows = [int(reverse_map(config["tag_map"])[tag]) for tag in item["ground_truth"]]
			for row, column in zip(rows, columns):
				con_matrix[row][column] += 1
			if columns != rows:
				mislabeled.append(item)
			else:
				correct.append(item)

		save_mislabeled(mislabeled)
		save_correct(correct)

	logging.info("Model evaluates {} transactions per second".format(total_trans / elapsed_time))
	con_matrix = pd.DataFrame(con_matrix)
	con_matrix.columns = [config["tag_map"][str(i)] for i in range(num_labels)]
	con_matrix_path = "data/RNN_stats/confusion_matrix.csv"
	con_matrix.to_csv(con_matrix_path, index=False)
	logging.info("Confusion matrix saved to: {0}".format(con_matrix_path))
	get_classification_report(con_matrix_path, config["tag_map"],
		"data/RNN_stats/classification_report.csv")

if __name__ == "__main__":
	evaluate_model()
