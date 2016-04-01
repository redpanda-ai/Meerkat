"""Fixtures for test_tools"""

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
