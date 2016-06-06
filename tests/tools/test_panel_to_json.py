"""Unit test for meerkat.panel_to_json"""

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

if __name__ == '__main__':
	unittest.main()
