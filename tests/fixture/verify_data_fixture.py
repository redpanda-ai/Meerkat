"""Fixtures for test_verify_data"""

def get_json_input_path():
	"""Return json file paths"""
	return {
		"correct_format": "meerkat/web_service/example_input.json",
		"mal_format": "tests/fixture/mal_format_json",
		"dup_key": "tests/fixture/dup_key.json",
		"not_found": "tests/missing.json"
	}

def get_csv_input_path():
	"""Return csv file paths"""
	return {
		"correct_format": "tests/fixture/correct_format.csv",
		"mal_format": "tests/fixture/mal_format.csv"
	}
