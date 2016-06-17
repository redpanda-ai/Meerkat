"""Fixtures for test_tensorflow_cnn"""

import numpy as np
import os.path

from meerkat.various_tools import load_params, load_piped_dataframe
from meerkat.classification.auto_load import main_program as load_models_from_s3

def get_batch():
	"""Return a batch of csv data"""
	return load_piped_dataframe("tests/classification/fixture/batch_to_tensor.csv")

def get_config_for_batch_to_tensor():
	"""Return a config dictionary for batch_to_tensor"""
	return {
		"doc_length": 3,
		"alphabet": "abc",
		"alphabet_length": 3,
		"num_labels": 2,
		"alpha_dict": {'a': 0, 'b': 1, 'c': 2}
	}

def get_trans_and_labels():
	"""Return result for batch_to_tensor"""
	labels = np.array([[1., 0.], [0., 1.]])
	trans = np.zeros((2, 1, 3, 3))
	for i in range(3):
		trans[0][0][i][2 - i] = 1
	trans[1][0][:,0] = 1
	return trans, labels

def get_config():
	"""Return a config dictionary"""
	alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
	alpha_dict = {a : i for i, a in enumerate(alphabet)}
	return {
		"alphabet": alphabet,
		"alpha_dict": alpha_dict
	}

def get_subtype_config():
	"""Return a different config dictionary"""
	label_map = "meerkat/classification/label_maps/subtype.card.credit.json"
	if not os.path.isfile(label_map):
		load_models_from_s3(prefix="meerkat/cnn/data/subtype/card/credit")
	return {
		"model_type": "subtype",
		"dataset": "tests/fixture/correct_format.csv",
		"label_map": load_params(label_map),
		"ledger_entry": "credit"
	}

def get_tensor(case_type):
	"""Return a tensor coverted from doc string"""
	if case_type == "short_doc":
		tensor = np.zeros((68, 4), dtype=np.float32)
		tensor[0][1] = 1.
		tensor[0][2] = 1.
		tensor[1][0] = 1.
	else:
		tensor = np.zeros((68, 2), dtype=np.float32)
		tensor[0] = [1., 1.]
	return tensor

