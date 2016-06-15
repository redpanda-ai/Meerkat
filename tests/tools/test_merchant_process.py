"""Unit test for meerkat.tools.merchant_process"""

import json
import unittest
import meerkat.tools.merchant_process as mp
from nose_parameterized import parameterized

class MerchantProcessTests(unittest.TestCase):
	"""Unit test for merchant_process.py"""

	def test_dict_2_json(self):
		dict_example = {'a': 1, 'bb': {'ccc': 222, 'ddd': 333}}
		mp.dict_2_json(dict_example, 'dict_example.json')

		json_obj = open('dict_example.json')
		json_str = json_obj.read()
		dict_expected = json.loads(json_str)

		self.assertEqual(dict_example, dict_expected)

if __name__ == '__main__':
	unittest.main()
