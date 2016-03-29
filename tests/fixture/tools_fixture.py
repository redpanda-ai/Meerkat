"""Fixtures for test_tools"""

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
		"with_empty_transaction": "tests/fixture/with_empty_transaction.csv"
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
