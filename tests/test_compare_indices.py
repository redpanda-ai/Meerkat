"""Unit test for meerkat.labeling_tools.compare_indices"""

import sys
import unittest
import meerkat.labeling_tools.compare_indices as compare_indices
from nose_parameterized import parameterized

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

if __name__ == '__main__':
	unittest.main()
