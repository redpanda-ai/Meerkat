"""Unit test for meerkat.various_tools"""

import unittest
from nose_parameterized import parameterized
import meerkat.bulk_loader as bl
from meerkat.custom_exceptions import InvalidArguments, Misconfiguration
import sys

class BulkLoaderTests(unittest.TestCase):
	"""UnitTest class for various_tools."""

	@parameterized.expand([
		([InvalidArguments, "single argument", []]),
		([IOError, "single argument", ["fake_file"]]),
	])
	def test_initialize(self, result, expected, system_arguments):
		"""Test initialize"""
		sys.argv = system_arguments
		if isinstance(result, Exception):
			self.assertRaisesRegex(result, expected, bl.initialize)

if __name__ == '__main__':
	unittest.main()
