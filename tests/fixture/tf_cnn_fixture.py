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
