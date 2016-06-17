"""Fixture for test_soft_target module"""

import numpy as np

def get_models():
	"""return 3 classifier that generates soft target"""
	def mock_classifier(trans, doc_key="description", soft_target=False):
		return np.array([[1, 1, 1],[1, 1, 1]])
	return [mock_classifier for i in range(3)]

def get_data():
	"""returns a fixture csv file that contains 2 transations"""
	return "tests/classification/fixture/with_empty_transaction.csv"
