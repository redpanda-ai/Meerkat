"""Unit test for meerkat/classification/tools.py"""

import os
import unittest
import csv
import pandas as pd
import meerkat.classification.tools as tools
from nose_parameterized import parameterized
from tests.fixture import tools_fixture
from plumbum import local

class ToolsTests(unittest.TestCase):
	"""Our UnitTest class."""

	@parameterized.expand([
		([tools_fixture.get_csvs_directory(), 2])
	])
	def test_merge_csvs(self, directory, merged_df_len):
		"""Test merge_csvs with parameters"""
		self.assertEqual(len(tools.merge_csvs(directory)), merged_df_len)

	@parameterized.expand([
		([tools_fixture.get_csv_path("correct_format"), "credit"])
	])
	def test_load(self, input_file, credit_or_debit):
		"""Test load with parameters"""
		results = tools.load(input_file=input_file, credit_or_debit=credit_or_debit)
		self.assertEqual(len(results[0]), 3)
		self.assertEqual(sorted(results[1]), sorted(["Other Income - Credit",
			"Deposits & Credits - Rewards", "Deposits & Credits - Deposit"]))

	@parameterized.expand([
		([["purchase", "rewards"], {"purchase": 1, "rewards": 2}]),
		([[], {}])
	])
	def test_get_label_map(self, class_names, result):
		"""Test get_label_map with parameters"""
		self.assertEqual(tools.get_label_map(class_names), result)

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
		(["with_file_name", {"bucket": "s3yodlee", "prefix": "Meerkat_tests_fixture", "extension": "csv",
			"file_name": "csv_file_1.csv", "save_path": "tests/fixture/"}]),
		(["found_multiple_files", {"bucket": "s3yodlee", "prefix": "Meerkat_tests_fixture", "extension": "csv",
			"save_path": "tests/fixture/"}]),
		(["file_not_found", {"bucket": "s3yodlee", "prefix": "Meerkat_tests_fixture", "extension": "csv",
			"file_name": "missing.csv", "save_path": "tests/fixture/"}])
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

	@parameterized.expand([
		(["tests/fixture/correct_format.csv", (3, 1)])
	])
	def test_seperate_debit_credit(self, subtype_file, output):
		"""Test correct_format with parameters"""
		result = tools.seperate_debit_credit(subtype_file)
		self.assertEqual((len(result[0]), len(result[1])), output)

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

	@parameterized.expand([
		(["purchase - purchase", "Purchase - Purchase"]),
		(["deposits - investment income or cash", "Deposits - Investment Income or Cash"])
	])
	def test_cap_first_letter(self, label, output):
		"""Test cap_first_letter with parameters"""
		self.assertEqual(tools.cap_first_letter(label), output)

if __name__ == '__main__':
	unittest.main()
