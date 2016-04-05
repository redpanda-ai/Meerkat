"""Fixtures for test_tensorflow_cnn"""

import numpy as np

def get_predictions(case_type):
	"""Return a numpy array of predictions"""
	np_array_all_correct = np.arange(4).reshape(2, 2)

	np_array_all_wrong = np.arange(4).reshape(2, 2)
	np_array_all_wrong[:, 0] = 4

	np_array_half_correct = np.arange(4).reshape(2, 2)
	np_array_half_correct[0, 0] = 4

	np_arrays = {
		"all_correct": np_array_all_correct,
		"all_wrong": np_array_all_wrong,
		"half_correct": np_array_half_correct
	}
	return np_arrays[case_type]

def get_labels():
	"""Return a numpy array of labels"""
	return np.arange(4).reshape(2,2)

def get_config():
	"""Return a config dictionary"""
	alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
	alpha_dict = {a : i for i, a in enumerate(alphabet)}
	return {
		"alphabet": alphabet,
		"alpha_dict": alpha_dict
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

