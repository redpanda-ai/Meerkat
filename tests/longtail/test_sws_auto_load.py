"""Unit test for meerkat/longtail/sws_auto_load.py"""

import unittest
from nose_parameterized import parameterized
import meerkat.longtail.sws_auto_load as load

class SwsAutoLoadTests(unittest.TestCase):
	"""Unit tests for meerkat/longtail/sws_auto_load.py """

	@parameterized.expand([
		(False, ["--bucket", "yodjie", "--prefix", "meerkat", "--save_path", "sws"])
	])
	def test_parse_arguments(self, exception_test, arguments):
		"""Test for parse_arguments"""
		parser = load.parse_arguments(arguments)
		if not exception_test:
			self.assertEqual(parser.bucket, "yodjie")
			self.assertEqual(parser.prefix, "meerkat")
			self.assertEqual(parser.save_path, "sws")

if __name__ == '__main__':
	unittest.main()
