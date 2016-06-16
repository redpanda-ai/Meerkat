"""Fixtures for test_tensorflow_cnn"""

import numpy as np
from meerkat.various_tools import load_params
from meerkat.classification.auto_load import main_program as load_models_from_s3
import os.path

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

