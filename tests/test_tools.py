"""Unit test for meerkat/classification/tools.py"""

import unittest
import pandas as pd
import meerkat.classification.tools as tools
from nose_parameterized import parameterized
from tests.fixture import tools_fixture
from plumbum import local

class ToolsTests(unittest.TestCase):
	"""Our UnitTest class."""

	@parameterized.expand([
		(["with_file_name", {"bucket": "s3yodlee", "prefix": "Meerkat_tests_fixture",
			"extension": "csv", "file_name": "csv_file_1.csv", "save_path": "tests/fixture/"}]),
		(["found_multiple_files", {"bucket": "s3yodlee", "prefix": "Meerkat_tests_fixture",
			"extension": "csv", "save_path": "tests/fixture/"}]),
		(["file_not_found", {"bucket": "s3yodlee", "prefix": "Meerkat_tests_fixture",
			"extension": "csv", "file_name": "missing.csv", "save_path": "tests/fixture/"}])
	])
	def test_pull_from_s3(self, case_type, inputs):
		"""Test pull_from_s3 with parameters"""
		kwargs = inputs
		if case_type == "found_multiple_files":
			self.assertRaises(Exception, tools.pull_from_s3, **kwargs)
		elif case_type == "file_not_found":
			self.assertRaises(Exception, tools.pull_from_s3, **kwargs)
		else:
			self.assertEqual(tools.pull_from_s3(**kwargs), "tests/fixture/csv_file_1.csv")
			local["rm"]["tests/fixture/csv_file_1.csv"]()

	@parameterized.expand([
		([tools_fixture.get_dict(), tools_fixture.get_reversed_dict()])
	])
	def test_reverse_map(self, label_map, output):
		"""Test reverse_map with parameters"""
		self.assertEqual(tools.reverse_map(label_map), output)

	@parameterized.expand([
		([pd.Series(["", "masked"], ["DESCRIPTION_UNMASKED", "DESCRIPTION"]), "masked"]),
		([pd.Series(["unmasked", "masked"], ["DESCRIPTION_UNMASKED", "DESCRIPTION"]), "unmasked"]),
		([pd.Series(["", ""], ["DESCRIPTION_UNMASKED", "DESCRIPTION"]), ""]),
	])
	def test_fill_description_unmasked(self, row, output):
		"""Test fill_description_unmasked with parameters"""
		self.assertEqual(tools.fill_description_unmasked(row), output)

if __name__ == '__main__':
	unittest.main()
