"""Unit test for meerkat/longtail/rnn_auto_load.py"""

import os
import unittest
from nose_parameterized import parameterized
import meerkat.longtail.rnn_auto_load as load

class RnnAutoLoadTests(unittest.TestCase):
	"""Unit tests for meerkat/longtail/rnn_auto_load.py """

	@parameterized.expand([
		(False, ["--bucket", "yodjie", "--prefix", "meerkat", "--save_path", "rnn"])
	])
	def test_parse_arguments(self, exception_test, arguments):
		"""Test for parse_arguments"""
		parser = load.parse_arguments(arguments)
		if not exception_test:
			self.assertEqual(parser.bucket, "yodjie")
			self.assertEqual(parser.prefix, "meerkat")
			self.assertEqual(parser.save_path, "rnn")

	@parameterized.expand([
		("s3yodlee", "meerkat/rnn/data/20160824/classification_report.csv")
	])
	def test_get_accuracy(self, bucket, file_path):
		"""Test for get_classification_accuracy"""
		accuracy = load.get_classification_accuracy(bucket, file_path)
		self.assertEqual(str(accuracy), "0.991893047916")

	@parameterized.expand([
		("s3yodlee", "meerkat/rnn/data/", "classification.csv", "gz.tar.results")
	])
	def test_get_valid_directories(self, bucket, prefix, report, model):
		"""Test for get_valid_model_directories"""
		directories = load.get_valid_model_directories(bucket, prefix, report, model)
		self.assertEqual(directories, [])

	@parameterized.expand([
		("s3yodlee", "meerkat/rnn/data/20160824/classification_report.csv", "./classification.csv")
	])
	def test_download_file_from_s3(self, bucket, source, destination):
		"""Test for download_file_from_s3"""
		load.download_file_from_s3(bucket, source, destination)
		self.assertEqual(os.path.exists(destination), 1)
		os.remove(destination)

if __name__ == '__main__':
	unittest.main()
