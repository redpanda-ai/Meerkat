"""Fixtures for test_verify_data"""

def get_json_input_path():
	"""Return json file paths"""
	return {
		"correct_format": "tests/fixture/correct_format.json",
		"dup_key": "dup_key.json",
		"not_found": "tests/missing.json"
	}

def get_csv_input_path():
	"""Return csv file paths"""
	return {
		"correct_format": "tests/fixture/correct_format.csv",
		"mal_format": "tests/classification/fixture/mal_format.csv"
	}

def get_csvs_directory():
	"""Return a directory contains several csv files"""
	return "tests/fixture/csvs/"
