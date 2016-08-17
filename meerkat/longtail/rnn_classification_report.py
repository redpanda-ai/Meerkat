#/usr/local/bin/python3.3

"""This module loads and evaluates a trained RNN (biLSTM) on a provided test set.
It produces various stats and a confusion matrix for analysis

@author: Oscar Pan
"""

############################# USAGE #############################
# python3 -m meerkat.longtail.rnn_classificaiton_report \
# <path_to_data> <path_to_cktp_file> <path_to_w2i_file> \
# --config <optional_path_to_config_file>
#################################################################

import logging
import argparse
import sys
import os
import pandas as pd
import numpy as np

from meerkat.classification.load_model import get_tf_rnn_by_path
from meerkat.various_tools import load_params, load_piped_dataframe
from meerkat.classification.classification_report import get_classification_report, count_transactions
from meerkat.longtail.bilstm_tagger import validate_config
from meerkat.classification.tools import reverse_map

def parse_arguments(args):
	"""Create parser"""
	parser = argparse.ArgumentParser(description="Test a RNN against a dataset and\
		return performance statistices")
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
	tmp = [config["tag_map"][str(i)] for i in np.argmax(item["predicted"], 1)]
	target_indices = [i for i in range(len(tmp)) if tmp[i] == "merchant"]
	tran = item["Description"].split()[:config["max_tokens"]]
	item["predicted"] = " ".join([tran[i] for i in target_indices])
	return item

def write_mislabeled(data, mode, config):
	if len(data) > 0:
		data = [beautify(item, config) for item in data]
		file_path = "./data/RNN_stats/mislabeled.csv"
		logging.info("Saving mislabeled transactions to {0}".format(file_path))
		df = pd.DataFrame(data)
		df.to_csv(file_path, sep="|", mode=mode, index=False)

def evaluate_model(args=None):
	os.makedirs("./data/RNN_stats/", exist_ok=True)
	if args is None:
		args = parse_arguments(sys.argv[1:])
	config = validate_config(args.config)
	num_labels = len(config["tag_map"])
	con_matrix = [[0] * num_labels for i in range(num_labels)]
	log_format = "%(asctime)s %(levelname)s: %(message)s"
	reader = pd.read_csv(args.data, sep="|", chunksize=1000)
	model = get_tf_rnn_by_path(args.model, args.w2i)
	total_trans = count_transactions(args.data)
	processed = 0.0
	mode = "w"
	for chunk in reader:
		processed += len(chunk)
		mislabeled = []
		logging.info("Processing {0:3.2f}% of the data...".format(100*processed/total_trans))
		chunk = chunk.to_dict("record")
		chunk = model(chunk, only_merchant_name=False, tags=True)
		for item in chunk:
			columns = list(np.argmax(item["predicted"], 1))
			rows = [int(reverse_map(config["tag_map"])[tag]) for tag in item["ground_truth"]]
			for i in range(len(rows)):
				con_matrix[rows[i]][columns[i]] += 1
			if columns != rows:
				mislabeled.append(item)
		write_mislabeled(mislabeled, mode, config)
		mode = "a"
	con_matrix = pd.DataFrame(con_matrix)
	con_matrix.columns = [config["tag_map"][str(i)] for i in range(num_labels)]
	con_matrix_path = "data/RNN_stats/confusion_matrix.csv"
	con_matrix.to_csv(con_matrix_path, index=False)
	logging.info("Confusion matrix saved to: {0}".format(con_matrix_path))
	get_classification_report(con_matrix_path, config["tag_map"], "data/RNN_stats/classification_report.csv")

if __name__ == "__main__":
	evaluate_model()
