"""Fixtures for test_get_merchants_by_id"""

def get_args():
	"""Return different arguments"""
	return {
		"not_enough": ["arg_0", "arg_1"],
		"no_json": ["arg_0", "arg_1.notjson"],
		"no_txt": ["arg_0", "arg_1.json", "arg_2.nottxt"],
		"not_correct": ["arg_0", "arg_1.json", "arg_2.nottxt"],
		"correct": ["arg_0", "arg_1.json", "arg_2.txt"]
	}
