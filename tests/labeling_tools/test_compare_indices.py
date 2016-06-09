"""Unit test for meerkat.labeling_tools.compare_indices"""

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

	@parameterized.expand([
		([[8.0, 4.0, 2.0, 1.0], 1.492]),
		([[2.0, 1.0], 2.0]),
		([[1.0], None])
	])
	def test_z_score_delta(self, scores, z_score_delta):
		self.assertEqual(compare_indices.z_score_delta(scores), z_score_delta)

if __name__ == '__main__':
	unittest.main()
