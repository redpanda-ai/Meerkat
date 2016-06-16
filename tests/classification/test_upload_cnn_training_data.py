"""Unit test for meerkat/classification/upload_cnn_training_data.py"""

#import os
import sys
import unittest
import meerkat.classification.upload_cnn_training_data as uploader

from nose_parameterized import parameterized
#from tests.classification.fixture import tools_fixture

class UploaderTests(unittest.TestCase):
	"""Our UnitTest class."""

	@parameterized.expand([
		([["", "", "merchant_bank"], "meerkat/cnn/data/merchant/bank/"])
	])
	def test_get_prefix(self, args, expected):
		"""Test get_prefix with parameters"""
		sys.argv = args
		self.assertEqual(uploader.get_prefix(), expected)

	@parameterized.expand([
		([["", "valid"], False]),
		([["", "no_json"], True]),
		([["", "two_jsons"], True]),
		([["", "no_csv"], True]),
		([["", "invalid_file_type"], True])
	])
	def test_check_file_existence(self, args, exception):
		"""Test check_file_existence with parameters"""
		fixture_dir = "tests/classification/fixture/fixture_for_upload_cnn_training_data/"
		args[1] = fixture_dir + args[1]
		sys.argv = args
		if exception:
			self.assertRaises(SystemExit, uploader.check_file_existence)
		else:
			uploader.check_file_existence()
