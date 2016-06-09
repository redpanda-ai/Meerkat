"""Unit test for meerkat/classification/tools.py"""

import meerkat.classification.classification_report as cr
import unittest
import argparse

from nose_parameterized import parameterized
from tests.classification.fixture import classification_report_fixture

class ClassifcationReportTests(unittest.TestCase):
	"""Unit tests for meerkat.classification.auto_load."""
	@parameterized.expand([
		(False, ["model", "data", "label_map", "label",  "--doc_key", "SNOZ"])
	])
	def test_parse_arguments(self, exception_test, arguments):
		"""Simple test to ensure that this function works"""
		if not exception_test:
			parser = cr.parse_arguments(arguments)
			self.assertEqual(parser.model, "model")
			self.assertEqual(parser.data, "data")
			self.assertEqual(parser.label_map, "label_map")
			self.assertEqual(parser.label, "label".upper())
			self.assertEqual(parser.doc_key, "SNOZ")

	@parameterized.expand([
		([classification_report_fixture.get_csv(), 18])
	])
	def test_count_transactions(self, filename, num_of_trans):
		"""Test count_transactions with parameters"""
		self.assertEqual(cr.count_transactions(filename), num_of_trans)

if __name__ == '__main__':
	unittest.main()
