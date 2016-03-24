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

def get_csv_path(input_type):
	"""Return a pandas dataframe"""
	csv_input = {
		"correct_format": "tests/fixture/correct_format.csv",
		"with_empty_transaction": "tests/fixture/with_empty_transaction.csv",
	}
	return csv_input[input_type]

def get_csvs_directory():
	"""Return a directory contains several csv files"""
	return "tests/fixture/csvs/"
