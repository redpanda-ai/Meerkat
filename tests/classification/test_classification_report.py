"""Unit test for meerkat/classification/tools.py"""

import argparse
import meerkat.classification.classification_report as cr
import unittest

from nose_parameterized import parameterized

class ClassifcationReportTests(unittest.TestCase):
	"""Unit tests for meerkat.classification.auto_load."""
	@parameterized.expand([
		(False, ["--model", "foo", "--label_map", "bar", "--testdata",
			"baz", "--label_key", "SNOZ"])
	])
	def test_parse_arguments(self, exception_test, arguments):
		"""Simple test to ensure that this function works"""
		if not exception_test:
			parser = cr.parse_arguments(arguments)
			self.assertEqual(parser.model, "foo")

if __name__ == '__main__':
	unittest.main()
