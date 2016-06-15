"""Unit test for meerkat.panel_to_json"""

import os
import json
import unittest
import meerkat.tools.panel_to_json as panel_to_json
from nose_parameterized import parameterized

class PaneltoJsonTests(unittest.TestCase):
	"""Our UnitTest class."""

	@parameterized.expand([
		(["ab'cd", "ab cd"]),
		(["ab*<cd", "ab cd"])
	])
	def test_string_cleanse(self, original_string, result):
		"""Test for string_cleanse"""
		self.assertEqual(panel_to_json.string_cleanse(original_string), result)

	def test_dict_2_json(self):
		"""Test dict_2_json"""
		dict_example = {'a': 1, 'bb': {'ccc': 222, 'ddd': 333}}
		filename = 'tests/tools/fixture/dict_example.json'
		panel_to_json.dict_2_json(dict_example, filename)

		json_obj = open(filename)
		json_str = json_obj.read()
		dict_expected = json.loads(json_str)

		self.assertEqual(dict_example, dict_expected)
		os.remove(filename)

if __name__ == '__main__':
	unittest.main()
