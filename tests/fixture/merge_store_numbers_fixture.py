"""Fixtures for test_merge_store_numbers"""

def get_csv_file():
	"""Return a csv file path"""
	return "tests/fixture/store_example.csv"

def get_args():
	"""Return a list of arguments"""
	return {
		"insufficient_arg": ["arg0"],
		"single_merchant": ["arg0", "tests/fixture/correct_format.csv", "arg2", "arg3", "arg4"],
		"directory_of_merchant": ["arg0", "tests/fixture/csvs/", "arg2", "arg3", "arg4"],
		"not_a_directory": ["arg0", "tests/fixture/missing", "arg2", "arg3", "arg4"],
		"no_csv": ["arg0", "tests/", "arg2", "arg3", "arg4"]
	}
