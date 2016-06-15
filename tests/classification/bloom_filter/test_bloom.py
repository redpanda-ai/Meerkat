'''unit tests for meerkat.classification.bloom_filter.bloom'''

import unittest

from nose_parameterized import parameterized
import meerkat.classification.bloom_filter.bloom as bloom

class BloomTests(unittest.TestCase):

	@parameterized.expand([
		([("TORONTO", "ZZ"), False]),
		([("DALLAS", "TX"), True])
	])
	def test_bloom_filter(self, location, expect):
		sbf = bloom.get_location_bloom()
		result = location in sbf
		self.assertEqual(expect, result)

	def test_get_json_from_file_exist(self):
		filename = 'tests/classification/bloom_filter/fixture/get_json.json'
		result = bloom.get_json_from_file(filename)
		self.assertEqual(isinstance(result, dict), True)

	def test_get_json_from_file_not_exist(self):
		filename = 'tests/classification/bloom_filter/fixture/not_exist.json'
		self.assertRaises(SystemExit, bloom.get_json_from_file, filename)

if __name__ == "__main__":
	unittest.main()
