"""Unit test for meerkat/longtail/cnn_sws.py"""

import unittest
import pandas as pd
import meerkat.longtail.sws as sws
import jsonschema as js
import numpy as np

from meerkat.various_tools import load_params
from nose_parameterized import parameterized

class CnnSwsTest(unittest.TestCase):
	"""Unit tests for meerkat.longtail.cnn_sws.py"""

	@parameterized.expand([
		([['invalid', 'tests/longtail/fixture/missing_entry_sws_config.json']]),
		([['valid', 'tests/longtail/fixture/valid_sws_config.json']]),
		([['invalid', 'tests/longtail/fixture/wrong_doc_length_sws_config.json']])
	])
	def test_validate_config(self, config):
		"""Confirm that the validate_config functions as expected."""
		case, config = config[:]
		if case == "invalid":
			self.assertRaises(js.exceptions.ValidationError, sws.validate_config, config)
		elif case == "valid":
			self.assertEqual(type(sws.validate_config(config)), dict)

	def test_load_data(self):
		"""Confirm that each transaction will have a new label 1 or 2""" 
		config = {"dataset": "tests/longtail/fixture/sample_data.csv",
			"label_name": "Tagged_merchant_string"
			}
		train, test, groups_train = sws.load_data(config)
		self.assertEqual(set(["1", "2"]), set(groups_train.keys()))
