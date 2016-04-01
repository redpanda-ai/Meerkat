"""Unit test for meerkat/classification/tools.py"""

import unittest
import meerkat.classification.tools as tools
from nose_parameterized import parameterized

class ToolsTests(unittest.TestCase):
	"""Our UnitTest class."""

	@parameterized.expand([
		(["input file not exist", {"bucket": "s3yodlee", "prefix": "Meerkat_tests_fixture/"}]),
		(["input file not exist", {"bucket": "s3yodlee", "prefix": "Meerkat_tests_fixture"}])
	])
	def test_pull_from_s3(self, case_type, s3params):
		"""Test check_new_input_file"""
		if case_type == "input file not exist":
			self.assertRaises(SystemExit, tools.check_new_input_file, **s3params)
		#else:
		#	self.assertEqual(tools.pull_from_s3(**inputs), "tests/fixture/csv_file_1.csv")
		#	local["rm"]["tests/fixture/csv_file_1.csv"]()

