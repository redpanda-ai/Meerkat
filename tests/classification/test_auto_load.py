"""Unit test for meerkat/classification/tools.py"""

import boto
import meerkat.classification.auto_load as auto_load
import os
import unittest

from nose_parameterized import parameterized
from os.path import isfile

def remove_file_if_exists(path):
	"""Helper function"""
	try:
		os.remove(path)
	except FileNotFoundError:
		pass

class AutoLoadTests(unittest.TestCase):
	"""Unit tests for meerkat.classification.auto_load."""

	@parameterized.expand([
		(["us-west-2", "s3yodlee", "meerkat/cnn/data", "results.tar.gz"]),
		(["us-west-2", "s3yodlee", "meerkat/cnn/data", "input.tar.gz"])
	])
	def test_find_s3_objects_recursively(self, region, bucket, prefix, target):
		"""Test the ability to find all candidate models in S3"""
		conn = boto.s3.connect_to_region(region)
		bucket = conn.get_bucket(bucket)
		my_results = {}
		auto_load.find_s3_objects_recursively(conn, bucket, my_results,
			prefix=prefix, target=target)
		self.assertTrue(my_results)

	@parameterized.expand([
		(["prefix", "/model_type/", ["20160413000000/", "20160413000001/"],
			{"/model_type/": ["20160413000000/", "20160413000001/"]}]),
		(["prefix", "/model_type/", ["20160413000000/"],
			{"/model_type/": ["20160413000000/"]}]),
		(["prefix", "/model_type/", [], {}])
	])
	def test_get_peer_models(self, prefix, model_type, timestamps, expected):
		"""Test the ability to get peer models from S3"""
		candidate_dictionary = {}
		for stamp in timestamps:
			candidate_dictionary[prefix + model_type + stamp] = "results.tar.gz"
		result = auto_load.get_peer_models(candidate_dictionary, prefix=prefix)
		self.assertDictEqual(result, expected)

	@parameterized.expand([
		(["con_matrix_1.csv", 0.99420625724217848]),
		(["con_matrix_2.csv", 0.5])
	])
	def test_get_model_accuracy(self, confusion_matrix, expected):
		"""Test the correct calculation of accuracy from the confusion matrix provided."""
		confusion_matrix = "tests/classification/fixture/" + confusion_matrix
		result = auto_load.get_model_accuracy(confusion_matrix)
		self.assertEqual(result, expected)

	@parameterized.expand([
		(["tarball_4.tar.gz", "tests/classification/fixture/"] )
	])
	def test_set_label_map(self, tarball, output_path):
		"""Test extraction of label_map from tarball."""
		tarball = output_path + tarball
		label_map = output_path + "label_map.json"
		remove_file_if_exists(label_map)
		result = auto_load.set_label_map(None, None, "/label_map/", None, tarball, output_path)
		self.assertTrue(isfile(result))
		remove_file_if_exists(label_map)

	@parameterized.expand([
		(["tarball_1.tar.gz", ".*txt", True, "not a tarfile."]),
		(["tarball_2.tar.gz", ".*csv", True, "several"]),
		(["tarball_3.tar.gz", "foo", False, "foo.txt"])
	])
	def test_get_single_file_from_tarball(self, tarball, pattern, exception, expected):
		"""Test to see that we can retrieve one and only one file from a tarball."""
		tarball = "tests/classification/fixture/" + tarball
		if exception:
			self.assertRaisesRegex(Exception, expected,
				auto_load.get_single_file_from_tarball, tarball, pattern)
		else:
			result = auto_load.get_single_file_from_tarball(tarball, pattern)
			self.assertEqual(result, expected)
if __name__ == '__main__':
	unittest.main()

