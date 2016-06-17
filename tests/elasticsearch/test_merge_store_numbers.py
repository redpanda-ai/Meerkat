"""Unit test for meerkat.merge_store_numbers"""

import sys
import unittest
import meerkat.elasticsearch.merge_store_numbers as merger
from nose_parameterized import parameterized
from tests.fixture import merge_store_numbers_fixture

class MergeStoreNumbersTests(unittest.TestCase):
	"""Our UnitTest class."""

	@parameterized.expand([
		(["insufficient_arg", merge_store_numbers_fixture.get_args()["insufficient_arg"]]),
		(["single_merchant", merge_store_numbers_fixture.get_args()["single_merchant"]]),
		(["directory_of_merchant", merge_store_numbers_fixture.get_args()["directory_of_merchant"]]),
		(["not_a_directory", merge_store_numbers_fixture.get_args()["not_a_directory"]]),
		(["no_csv", merge_store_numbers_fixture.get_args()["no_csv"]])
	])
	def test_verify_arguments(self, case_type, input_argv):
		"""Test verify_arguments with parameters"""
		sys.argv = input_argv
		if case_type == "insufficient_arg" or case_type == "not_a_directory" \
			or case_type == "no_csv":
			self.assertRaises(SystemExit, merger.verify_arguments)
		else:
			merger.verify_arguments()

	@parameterized.expand([
		([merge_store_numbers_fixture.get_csv_file()])
	])
	def test_load_store_numbers(self, file_name):
		"""Test load_store_numbers with parameters"""
		expected = [
			{'keywords': 'AutoZone', 'city': 'GAMBRILLS'},
			{'keywords': 'AutoZone', 'city': 'GAFFNEY'}
		]
		result = merger.load_store_numbers(file_name)
		self.assertEqual(result, expected)


	@parameterized.expand([
		([merge_store_numbers_fixture.get_result_dict()["empty_result"], 0, (False, False)]),
		([merge_store_numbers_fixture.get_result_dict()["normal_result"], 0, (2.0, "s0")])
	])
	def test_get_hit(self, search_results, index, result):
		"""Test get_hit with parameters"""
		self.assertEqual(merger.get_hit(search_results, index), result)

if __name__ == '__main__':
	unittest.main()
