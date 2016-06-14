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

if __name__ == "__main__":
	unittest.main()
