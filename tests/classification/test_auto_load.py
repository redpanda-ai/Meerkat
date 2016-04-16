"""Unit test for meerkat/classification/tools.py"""

import boto
import unittest
import meerkat.classification.auto_load as auto_load

from nose_parameterized import parameterized
from plumbum import local

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


if __name__ == '__main__':
	unittest.main()

