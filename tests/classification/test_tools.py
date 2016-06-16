"""Unit test for meerkat/classification/tools.py"""

import os
import csv
import unittest
import shutil
import pandas as pd
import meerkat.classification.tools as tools
from meerkat.various_tools import load_params
from nose_parameterized import parameterized
from tests.classification.fixture import tools_fixture

class ToolsTests(unittest.TestCase):
	"""Our UnitTest class."""

	@parameterized.expand([
		([[], 3, []]),
		([[1], 3, [[1]]]),
		([[1,2,3], 2, [[1,2],[3]]]),
		([[1,2], -9, [[1],[2]]])
	])
	def test_chunks(self, array, num, expected):
		"""Ensure that numpy arrays are properly sub-divided into chunks."""
		self.assertEqual(tools.chunks(array, num), expected)

	@parameterized.expand([
		([tools_fixture.get_output_filename(), tools_fixture.get_source_dir()])
	])
	def test_make_tarfile(self, output_filename, source_dir):
		"""Test make_tarfile with parameters"""
		tools.make_tarfile(output_filename, source_dir)
		self.assertTrue(os.path.isfile(output_filename))
		os.remove(output_filename)

	@parameterized.expand([
		(["invalid_tarfile", tools_fixture.get_archive_path("invalid_tarfile"),
			tools_fixture.get_des_path(), ""]),
		(["valid_tarfile", tools_fixture.get_archive_path("valid_tarfile"),
			tools_fixture.get_des_path(), "foo.txt"])
	])
	def test_extract_tarball(self, case_type, archive, des, file_name):
		"""Test extract_tarball with parameters"""
		if case_type == "invalid_tarfile":
			self.assertRaises(Exception, tools.extract_tarball, archive, des)
		else:
			tools.extract_tarball(archive, des)
			extracted_file = des + file_name
			self.assertTrue(os.path.isfile(extracted_file))
			os.remove(extracted_file)

	@parameterized.expand([
		(["foo.csv", tools_fixture.get_s3_params_to_check_file_existence(), True]),
		(["missing.csv", tools_fixture.get_s3_params_to_check_file_existence(), False])
	])
	def test_check_file_exist_in_s3(self, target_file_name, params, expected):
		"""Test check_file_exist_in_s3 with parameters"""
		self.assertEqual(tools.check_file_exist_in_s3(target_file_name, **params), expected)

	@parameterized.expand([
		(["with_file_name", tools_fixture.get_s3_params_to_pull_from_s3("with_file_name")]),
		(["found_multiple_files", tools_fixture.get_s3_params_to_pull_from_s3("found_multiple_files")]),
		(["file_not_found", tools_fixture.get_s3_params_to_pull_from_s3("file_not_found")])
	])
	def test_pull_from_s3(self, case_type, inputs):
		"""Test pull_from_s3 with parameters"""
		if case_type in ["found_multiple_files", "file_not_found"]:
			self.assertRaises(Exception, tools.pull_from_s3, **inputs)
		else:
			self.assertEqual(tools.pull_from_s3(**inputs), "tests/fixture/csv_file_1.csv")
			os.remove("tests/fixture/csv_file_1.csv")

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
			os.remove("empty_transactions.csv")

	@parameterized.expand([
		(["tests/fixture/correct_format.csv", "credit", "subtype", 3])
	])
	def test_seperate_debit_credit(self, subtype_file, credit_or_debit, model_type, output):
		"""Test correct_format with parameters"""
		result = tools.seperate_debit_credit(subtype_file, credit_or_debit, model_type)
		self.assertEqual(len(result), output)

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

	@parameterized.expand([
		([tools_fixture.get_predictions("all_correct"), tools_fixture.get_labels(), 100.0]),
		([tools_fixture.get_predictions("all_wrong"), tools_fixture.get_labels(), 0.0]),
		([tools_fixture.get_predictions("half_correct"), tools_fixture.get_labels(), 50.0])
	])
	def test_accuracy(self, predictions, labels, expected_accuracy):
		"""Test accuracy with parameters"""
		result = tools.accuracy(predictions, labels)
		self.assertEqual(result, expected_accuracy)

	@parameterized.expand([
		(["tests/classification/fixture/cost_matrix.json", [1.3, 0.8, 1.1]])
	])
	def test_get_cost_list(self, label_map_path, expected):
		"""Confirm that you can load a cost matrix from a proper label_map"""
		config = {"label_map": load_params(label_map_path)}
		results = tools.get_cost_list(config)
		self.assertTrue(sorted(results) == sorted(expected))

	@parameterized.expand([
		(["tests/fixture/csvs/", 2])
	])
	def test_merge_csvs(self, directory, expected_len):
		"""Test merge_csvs with parameters"""
		self.assertEqual(len(tools.merge_csvs(directory)), expected_len)

	@parameterized.expand([
		([tools_fixture.get_gz_file("no_json"), "card", (), True]),
		([tools_fixture.get_gz_file("two_jsons"), "card", (), True]),
		([tools_fixture.get_gz_file("valid"), "card",
			tools_fixture.get_unzip_and_merge_result(), False])
	])
	def test_unzip_and_merge(self, gz_file, bank_or_card, expected, exception):
		"""Test unzip_and_merge with parameters"""
		if exception:
			self.assertRaises(Exception, tools.unzip_and_merge, gz_file, bank_or_card)
		else:
			result = tools.unzip_and_merge(gz_file, bank_or_card)
			self.assertEqual(len(result[0]), expected[0])
			self.assertEqual(result[1], expected[1])
		shutil.rmtree("./merchant_card_unzip/")

	@parameterized.expand([
		(["ab cd", "Ab Cd"]),
		(["ab in cd", "Ab in Cd"])
	])
	def test_cap_first_letter(self, label, expected):
		"""Test cap_first_letter with parameters"""
		self.assertEqual(tools.cap_first_letter(label), expected)

	@parameterized.expand([
		([tools_fixture.get_dict(), "tests/classification/fixture/bar.json"])
	])
	def test_dict_2_json(self, obj, filename):
		"""Test dict_2_json with parameters"""
		tools.dict_2_json(obj, filename)
		self.assertTrue(os.path.isfile(filename))
		os.remove(filename)

	@parameterized.expand([
		(['tests/classification/fixture/tf_config_with_alpha_dict.json', 'kkk', 3, [[0,0,0],[0,0,0],[0,0,0]]]),
		(['tests/classification/fixture/tf_config_with_alpha_dict.json', 'abc', 3, [[0,0,1],[0,1,0],[1,0,0]]]),
		(['tests/classification/fixture/tf_config_with_alpha_dict.json', 'abccccccccc', 3, [[0,0,1],[0,1,0],[1,0,0]]]),
		(['tests/classification/fixture/tf_config_with_alpha_dict.json', 'a', 3, [[1,0,0],[0,0,0],[0,0,0]]])
	])
	def test_string_to_tensor_1(self, config, doc, length, expected):
		"""Test string_to_tensor with parameters"""
		config = load_params(config)
		np.testing.assert_array_equal(list(tools.string_to_tensor(config, doc, length)), expected)

	@parameterized.expand([
		([tools_fixture.get_config(), "aab", 4, tools_fixture.get_tensor("short_doc")]),
		([tools_fixture.get_config(), "aab", 2, tools_fixture.get_tensor("truncated_doc")])
	])
	def test_string_to_tensor_2(self, config, doc, length, expected_tensor):
		"""Test string_to_tensor with parameters"""
		result = tools.string_to_tensor(config, doc, length)
		np.testing.assert_array_equal(result, expected_tensor)

if __name__ == '__main__':
	unittest.main()

