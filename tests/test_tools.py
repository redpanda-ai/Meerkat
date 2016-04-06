"""Unit test for meerkat/classification/tools.py"""

import csv
import unittest
import pandas as pd
import meerkat.classification.tools as tools
from nose_parameterized import parameterized
from tests.fixture import tools_fixture
from plumbum import local

class ToolsTests(unittest.TestCase):
	"""Our UnitTest class."""

	@parameterized.expand([
		(["with_file_name", tools_fixture.get_s3_params("with_file_name")]),
		(["found_multiple_files", tools_fixture.get_s3_params("found_multiple_files")]),
		(["file_not_found", tools_fixture.get_s3_params("file_not_found")])
	])
	def test_pull_from_s3(self, case_type, inputs):
		"""Test pull_from_s3 with parameters"""
		if case_type == "found_multiple_files" or case_type == "file_not_found":
			self.assertRaises(Exception, tools.pull_from_s3, **inputs)
		else:
			self.assertEqual(tools.pull_from_s3(**inputs), "tests/fixture/csv_file_1.csv")
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
		([pd.Series(["", ""], ["DESCRIPTION_UNMASKED", "DESCRIPTION"]), ""])
	])
	def test_fill_description_unmasked(self, row, output):
		"""Test fill_description_unmasked with parameters"""
		self.assertEqual(tools.fill_description_unmasked(row), output)

	@parameterized.expand([
		(["non_empty", tools_fixture.get_csv_path("correct_format"), 4]),
		(["with_empty", tools_fixture.get_csv_path("with_empty_transaction"), 1])
	])
	def test_check_empty_transation(self, case_type, csv_path, output):
		"""Test check_empty_transaction with parameters"""
		df = pd.read_csv(csv_path, quoting=csv.QUOTE_NONE, na_filter=False,
			encoding="utf-8", sep='|', error_bad_lines=False)
		self.assertEqual(len(tools.check_empty_transaction(df)), output)
		if case_type == "with_empty":
			local["rm"]["empty_transactions.csv"]()

	@parameterized.expand([
		(["tests/fixture/correct_format.csv", (3, 1)])
	])
	def test_seperate_debit_credit(self, subtype_file, output):
		"""Test correct_format with parameters"""
		result = tools.seperate_debit_credit(subtype_file)
		self.assertEqual((len(result[0]), len(result[1])), output)

	@parameterized.expand([
		(["missing_input", tools_fixture.get_s3params("missing_input"),
			tools_fixture.get_result("missing_input")]),
		(["unpreprocessed", tools_fixture.get_s3params("unpreprocessed"),
			tools_fixture.get_result("unpreprocessed")]),
		(["preprocessed", tools_fixture.get_s3params("preprocessed"),
			tools_fixture.get_result("preprocessed")]),
		(["missing_slosh", tools_fixture.get_s3params("missing_slosh"),
			tools_fixture.get_result("missing_slosh")])
	])
	def test_check_new_input_file(self, case_type, s3params, result):
		"""Test check_new_input_file"""
		if case_type == "missing_input":
			self.assertRaises(SystemExit, tools.check_new_input_file, **s3params)
		else:
			self.assertEqual(tools.check_new_input_file(**s3params), result)

if __name__ == '__main__':
	unittest.main()

