"""Unit test for meerkat.various_tools"""

import unittest
import meerkat.various_tools as various_tools
from nose_parameterized import parameterized
from tests.fixture import various_tools_fixture

class VariousToolsTests(unittest.TestCase):

	"""Our UnitTest class."""

	strings = (("Weekend at Bernie's[]{}/:", "weekend at bernie s"),
                ('Trans with "" quotes', 'trans with quotes'))

	numbers = (("[8]6{7}5'3/0-9", "8675309"),
				('333"6"6"6"999', '333666999'))

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
		([various_tools_fixture.get_queue()["non_empty"], [1, 2, 3]]),
		([various_tools_fixture.get_queue()["empty"], []])
	])
	def test_queue_to_list(self, result_queue, result_list):
		"""Test queue_to_list with parameters"""
		self.assertEqual(various_tools.queue_to_list(result_queue), result_list)

	@parameterized.expand([
		(["abcd\\|ef\"ghij", "cd efg"]),
		(["abcd\\ef\"\"ghij", "cd\\efg"]),
		(["abcd|efghij", "cd|efg"])
	])
	def test_clean_line(self, line, output):
		"""Test clean_line with parameters"""
		self.assertEqual(various_tools.clean_line(line), output)

	@parameterized.expand([
		(["ach pos ", ""])
	])
	def test_stopwords(self, transaction, result):
		"""Test stopwords with parameters"""
		self.assertEqual(various_tools.stopwords(transaction), result)
if __name__ == '__main__':
	unittest.main()
