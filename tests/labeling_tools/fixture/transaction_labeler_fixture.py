"""Fixtures for test_transaction_labeler"""

def get_args():
	"""Return different arguments list"""
	return {
		"not_enough": ["arg_0"],
		"enough": ["arg_0", "arg_1"]
	}
