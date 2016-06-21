"""Unit test for meerkat.merge_store_numbers"""

import sys
import unittest
import meerkat.elasticsearch.load_index_from_file as loader
from nose_parameterized import parameterized

class LoadIndexFromFileTests (unittest.TestCase):
	"""Our UnitTest class."""

	@classmethod
	def setUpClass(cls):
		pass

	def test_foo(self):
		"""Sample unit test."""
		self.assertTrue(True)

	@parameterized.expand([
		(False, ["foo.json"])
	])
	def test_parse_arguments(self, exception_test, arguments):
		"""Simple test to ensure that this function works"""
		if not exception_test:
			parser = loader.parse_arguments(arguments)
			self.assertEqual(parser.configuration_file, arguments[0])

if __name__ == '__main__':
	unittest.main()
