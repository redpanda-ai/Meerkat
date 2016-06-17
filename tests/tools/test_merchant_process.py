"""Unit test for meerkat.tools.merchant_process"""

import os
import json
import unittest
import meerkat.tools.merchant_process as mp
from nose_parameterized import parameterized

class MerchantProcessTests(unittest.TestCase):
	"""Unit test for merchant_process.py"""

	def test_dict_2_json(self):
		"""Test dict_2_json"""
		dict_example = {'a': 1, 'bb': {'ccc': 222, 'ddd': 333}}
		filename = 'tests/tools/fixture/dict_example.json'
		mp.dict_2_json(dict_example, filename)

		json_obj = open(filename)
		json_str = json_obj.read()
		dict_expected = json.loads(json_str)

		self.assertEqual(dict_example, dict_expected)
		os.remove(filename)

if __name__ == '__main__':
	unittest.main()
