"""Unit test for meerkat/longtail/sws_auto_train.py"""

import unittest
from nose_parameterized import parameterized
import meerkat.longtail.sws_auto_train as train

class SwsAutoTrainTests(unittest.TestCase):
	"""Unit tests for meerkat/longtail/sws_auto_train.py """

	@parameterized.expand([
		(False, ["--bucket", "yodjie", "--prefix", "meerkat",\
				 "--config", "config.json", "--output_dir", "sws"])
	])
	def test_parse_arguments(self, exception_test, arguments):
		"""Test for parse_argument"""
		parser = train.parse_arguments(arguments)
		if not exception_test:
			self.assertEqual(parser.bucket, "yodjie")
			self.assertEqual(parser.prefix, "meerkat")
			self.assertEqual(parser.config, "config.json")
			self.assertEqual(parser.output_dir, "sws")

if __name__ == '__main__':
	unittest.main()

