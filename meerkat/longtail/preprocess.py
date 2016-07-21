#!/usr/local/bin/python3.3

"""This module loads and preprocess transacitons in accordance with the
NER model"""

########################## USAGE ######################################

#python3 -m meerkat.longtail_handler.preprocess meerkat/longtail_handler/csv_file

#######################################################################
import sys
import numpy as np

from meerkat.various_tools import load_piped_dataframe

def run_from_command_line(csv_path):
	"""Load and split data into training and test set"""
	data = load_piped_dataframe(csv_path)
	data = data[["Description", "Tagged_merchant_string"]]
	msk = np.random.rand(len(data)) < 0.90
	train = data[msk]
	test = data[~msk]

	base_path = "meerkat/longtail_handler/"
	train.to_csv(base_path+"train.csv", index=False, sep="|")
	test.to_csv(base_path+"test.csv", index=False, sep="|")

if __name__ == "__main__":
	run_from_command_line(sys.argv[1])
