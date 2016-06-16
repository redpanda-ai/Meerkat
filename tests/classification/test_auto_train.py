'''Unit tests for meerkat/classification/auto_train.py'''

import os
import unittest
import meerkat.classification.auto_load as auto_load
from nose_parameterized import parameterized

class AutoTrainTests(unittest.TestCase):

	@parameterized.expand([
		(False, ["model_type", "bank", "--credit_or_debit", 'debit'])
	])
	def test_parse_arguments(self, exception_test, arguments):
		"""Simple test to ensure that this function works"""
		if not exception_test:
			parser = auto_load.parse_arguments(arguments)
			self.assertEqual(parser.model_type, "model")
			self.assertEqual(parser.bank_or_card, "bank")
			self.assertEqual(parser.credit_or_debit, "debit")
