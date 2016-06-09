"""Unit test for meerkat.various_tools"""

import unittest
import meerkat.various_tools as various_tools
from nose_parameterized import parameterized
from tests.fixture import various_tools_fixture

class VariousToolsTests(unittest.TestCase):

	"""Our UnitTest class."""

	strings = (("Weekend at Bernie's[]{}/:", "weekend at bernie s"),
                ('Trans with "" quotes', 'trans with quotes'))

	def test_string_cleanse(self):
		"""string_cleanse test"""

		for chars, no_chars in self.strings:
			result = various_tools.string_cleanse(chars)
			self.assertEqual(no_chars, result)

	@parameterized.expand([
		(["normal", various_tools_fixture.get_params_dict()["correct_format"]]),
		(["edge", various_tools_fixture.get_params_dict()["not_found"]])
	])
	def test_load_hyperparameters(self, case_type, params):
		"""Test load_hyperparameters with parameters"""
		if case_type == "normal":
			result = various_tools.load_hyperparameters(params)
			self.assertTrue(isinstance(result, dict))
		else:
			self.assertRaises(SystemExit, various_tools.load_hyperparameters, params)

	@parameterized.expand([
		(["ach pos ", ""]),
		(["pos 12/34abcd", "ABCD"]),
		(["ab~~12345~~1234567890123456~~12345~~1~~~~1234cd", "AB CD"])
	])
	def test_stopwords(self, transaction, result):
		"""Test stopwords with parameters"""
		self.assertEqual(various_tools.stopwords(transaction), result)

	@parameterized.expand([
		([[{}, "Null", various_tools_fixture.get_es_connection("127.0.0.1")], None])
	])
	def test_get_merchant_by_id(self, args, result):
		"""Test get_merchant_by_id with parameters"""
		self.assertEqual(various_tools.get_merchant_by_id(*args), result)

if __name__ == '__main__':
	unittest.main()
