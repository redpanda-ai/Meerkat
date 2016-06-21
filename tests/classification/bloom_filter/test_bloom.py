'''unit tests for meerkat.classification.bloom_filter.bloom'''

import unittest
import os

from nose_parameterized import parameterized
import meerkat.classification.bloom_filter.bloom as bloom

class BloomTests(unittest.TestCase):

	@parameterized.expand([
		([("TORONTO", "ZZ"), False]),
		([("DALLAS", "TX"), True])
	])
	def test_bloom_filter(self, location, expected):
		sbf = bloom.get_location_bloom()
		result = location in sbf
		self.assertEqual(expected, result)

	def test_get_json_from_file_exist(self):
		filename = 'tests/classification/bloom_filter/fixture/get_json.json'
		result = bloom.get_json_from_file(filename)
		self.assertEqual(isinstance(result, dict), True)

	def test_get_json_from_file_not_exist(self):
		filename = 'tests/classification/bloom_filter/fixture/not_exist.json'
		self.assertRaises(SystemExit, bloom.get_json_from_file, filename)

	@parameterized.expand([
		(["tests/classification/bloom_filter/fixture/create_bloom.json", "tests/classification/bloom_filter/fixture/location_bloom", False, ("SANFRANCISCO", "CA"), True]),
	])
	def test_create_bloom(self, src_filename, dst_filename, enrich, case, expected):
		sbf = bloom.create_location_bloom(src_filename, dst_filename, False)
		result = case in sbf
		self.assertEqual(expected, result)
		os.remove(dst_filename)

if __name__ == "__main__":
	unittest.main()
