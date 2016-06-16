"""Unit test for meerkat.labeling_tools.compare_indices"""

import sys
import unittest
import meerkat.labeling_tools.compare_indices as compare_indices
from nose_parameterized import parameterized
from tests.labeling_tools.fixture import compare_indices_fixture

class CompareIndicesTests(unittest.TestCase):
	"""Our UnitTest class."""

	@parameterized.expand([
		([{"CITY": "la"}, {"CITY": "la"}, False]),
		([{"STATE": "CA"}, {"STATE": "WA"}, True]),
		([{"CITY": "la", "STATE": "CA"}, {"CITY": "la"}, True])
	])
	def test_has_mapping_changed(self, old_mapping, new_mapping, result):
		"""Test for has_mapping_changed"""
		self.assertEqual(compare_indices.has_mapping_changed(old_mapping, new_mapping), result)

	@parameterized.expand([
		([compare_indices_fixture.get_elasticsearch_result()["non_hits"], 0, (False, False)]),
		([compare_indices_fixture.get_elasticsearch_result()["has_hits"], 0, (2.0, "result_0")])
	])
	def test_get_hit(self, elasticsearch_result, index, expected_result):
		"""Test get_hit with parameters"""
		self.assertEqual(compare_indices.get_hit(elasticsearch_result, index), expected_result)

	@parameterized.expand([
		([compare_indices_fixture.get_transaction(), compare_indices_fixture.get_cleaned_transaction()])
	])
	def test_clean_transaction(self, transaction, cleaned_transaction):
		"""Test clean_transaction with parameters"""
		transaction = compare_indices_fixture.get_transaction()
		cleaned_transaction = compare_indices_fixture.get_cleaned_transaction()
		self.assertEqual(compare_indices.clean_transaction(transaction), cleaned_transaction)

if __name__ == '__main__':
	unittest.main()
