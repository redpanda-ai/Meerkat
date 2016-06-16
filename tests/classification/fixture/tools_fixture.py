"""Fixtures for test_tools"""
import numpy as np

def get_dict():
	"""Return a dictionary"""
	return {
		1: 0.0395,
		2: 0.0238,
		3: 0.0179,
		4: 0.0134
	}

def get_reversed_dict():
	"""Return a reversed dictionary"""
	return {
		0.0395: 1,
		0.0238: 2,
		0.0179: 3,
		0.0134: 4
	}

def get_csv_path(csv_type):
	"""Return a csv file path"""
	csv_path = {
		"correct_format": "tests/fixture/correct_format.csv",
		"with_empty_transaction": "tests/classification/fixture/with_empty_transaction.csv"
	}
	return csv_path[csv_type]

def get_s3_params(case_type):
	"""Return a dictionary of s3 params"""
	file_names = {
		"with_file_name": "csv_file_1.csv",
		"file_not_found": "missing.csv"
	}
	s3_params = {
			"bucket": "s3yodlee",
			"prefix": "Meerkat_tests_fixture",
			"extension": "csv",
			"save_path": "tests/fixture/"
		}
	if case_type == "with_file_name" or case_type == "file_not_found":
		s3_params["file_name"] = file_names[case_type]
	return s3_params

def get_s3params(case_type):
	"""Return a dictionary of s3 params"""
	prefix = {
		"missing_input": "Meerkat_tests_fixture/missing_input/",
		"unpreprocessed": "Meerkat_tests_fixture/unpreprocessed/",
		"preprocessed": "Meerkat_tests_fixture/preprocessed/",
		"missing_slosh": "Meerkat_tests_fixture/preprocessed"
	}
	return {
		"bucket": "s3yodlee",
		"prefix": prefix[case_type]
	}

def get_result(case_type):
	"""Return a tuple of result"""
	newest_version_dir_unprocessed = "Meerkat_tests_fixture/unpreprocessed/201604011500"
	newest_version_dir_processed = "Meerkat_tests_fixture/preprocessed/201604011500"
	newest_version = "201604011500"
	if case_type == "missing_input":
		return ()
	elif case_type == "unpreprocessed":
		return (True, newest_version_dir_unprocessed, newest_version)
	else:
		return (False, newest_version_dir_processed, newest_version)

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

