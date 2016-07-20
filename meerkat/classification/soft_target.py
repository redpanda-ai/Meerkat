#!/usr/local/bin/python3
# pylint: disable=pointless-string-statement

"""Produce soft target for training data

Created on May 17, 2016
@author: Oscar Pan
"""

############################################# USAGE ###############################################
"""
python3 -m meerkat.classification.soft_target \
<directory_containing_cnns> <training_data_path> <label_map_path>

python3 -m meerkat.classification.soft_target \
meerkat/classification/models/ensemble_cnns/ data/train.csv data/label.json
"""
###################################################################################################

import logging
import os
import sys
import pandas as pd

from meerkat.classification.load_model import get_tf_cnn_by_path
from meerkat.various_tools import load_piped_dataframe

def load_multiple_models(cnns_dir, label_map_path):
	"""load multiple models"""
	models = []
	for item in os.listdir(cnns_dir):
		if item.endswith(".ckpt"):
			model_name = item.split(".")[1]
			classifier = get_tf_cnn_by_path(cnns_dir + item, label_map_path,
				model_name=model_name+"/softmax_full:0")
			logging.info("Loaded model " + item)
			models.append(classifier)
	return models

def get_soft_target(data_path, models, output_path):
	"""load training data, produce soft target for each data"""
	logging.info("Producing soft target for training data")
	reader = load_piped_dataframe(data_path, chunksize=1000)
	os.makedirs(output_path, exist_ok=True)
	file_exist = False
	for trans in reader:
		trans = trans.to_dict("records")
		softmax = [classifier(trans, doc_key="DESCRIPTION_UNMASKED", soft_target=True)
			for classifier in models]
		ensemble_softmax = sum(softmax) / (len(softmax) + 0.0)
		num_labels = len(ensemble_softmax[0])
		for index, transaction in enumerate(trans):
			for i in range(1, num_labels+1):
				transaction["class_"+str(i)] = ensemble_softmax[index][i-1]

		mode = "a" if file_exist else "w"
		trans = pd.DataFrame(trans)
		add_header = False if file_exist else trans.columns
		logging.info("Saving soft targets to {0}".format(output_path + "soft_target.csv"))
		trans.to_csv(output_path + "soft_target.csv", mode=mode, index=False, header=add_header, sep='|')
		file_exist = True
	logging.info("Soft target production is finished.")
	return output_path + "soft_target.csv"

def main(cnns_dir, data_path, label_map_path):
	"""Process that produces soft targets"""
	output_path = "data/output/"
	models = load_multiple_models(cnns_dir, label_map_path)
	_ = get_soft_target(data_path, models, output_path)

if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2], sys.argv[3])


