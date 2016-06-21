"""Unit test for meerkat.various_tools"""

import sys
import unittest
from nose_parameterized import parameterized
import meerkat.various_tools as various_tools
from tests.fixture import various_tools_fixture

class VariousToolsTests(unittest.TestCase):
	"""UnitTest class for various_tools."""

	@parameterized.expand([
		(['term', None, { "query_string": { "query": 'term', "fields": [], "boost": 1.0 } }]),
		(['term', ['city'], { "query_string": { "query": 'term', "fields": ['city'], "boost": 1.0 } }])
	])
	def test_get_qs_query(self, term, field_list, expected):
		result = various_tools.get_qs_query(term, field_list)
		self.assertEqual(result, expected)

	def test_get_bool_query(self):
		result = various_tools.get_bool_query()
		self.assertEqual(result, { "from" : 0, "size" : 0, "query" : { "bool": { "minimum_number_should_match": 1, "should": [] } } })

	@parameterized.expand([
		([[8.0, 4.0, 2.0, 1.0], 1.492]),
		([[2.0, 1.0], 2.0]),
		([[1.0], None])
	])
	def test_z_score_delta(self, scores, z_score_delta):
		"""Test z_score_delta with parameters"""
		self.assertEqual(various_tools.z_score_delta(scores), z_score_delta)

	@parameterized.expand([
		(["Weekend at Bernie's[]{}/:", "weekend at bernie s"]),
		(['Trans with "" quotes', 'trans with quotes'])
	])
	def test_string_cleanse(self, input_str, expected_str):
		"""Test string_cleanse with parameters"""
		self.assertEqual(various_tools.string_cleanse(input_str), expected_str)

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

	@parameterized.expand([
		(["wal-mart", " WALMART "]),
		(["wal mart", " WALMART "]),
		(["samsclub", " SAM'S CLUB "]),
		(["usps", " US POST OFFICE "]),
		(["lowes", " LOWE'S "]),
		(["wholefds", " WHOLE FOODS "]),
		(["shell oil", " SHELL GAS "]),
		(["wm supercenter", " WALMART "]),
		(["exxonmobil", " EXXONMOBIL EXXON MOBIL "]),
		(["mcdonalds", " MCDONALD'S "]),
		(["costco whse", " COSTCO "]),
		(["franciscoca", " FRANCISCO CA "]),
		(["qt", " QUICKTRIP "]),
		(["macy's east", " MACY'S "])
	])
	def test_synonyms(self, input_str, expected_str):
		"""Test synonyms with parameters"""
		self.assertEqual(various_tools.synonyms(input_str), expected_str)

	def test_load_dict_list(self):
		"""Test load_dict_list"""
		dict_list = various_tools_fixture.get_dict_list()
		self.assertEqual(various_tools.load_dict_list("tests/fixture/dict_example.csv"), dict_list)

	def test_load_dict_ordered(self):
		"""Test load_dict_ordered"""
		expected_dict_list = various_tools_fixture.get_dict_list()
		expected_fieldnames = ['first_name,last_name,address,city,state,zip_code']

		dict_list, fieldnames = various_tools.load_dict_ordered("tests/fixture/dict_example.csv")

		self.assertEqual(expected_dict_list, dict_list)
		self.assertEqual(expected_fieldnames, fieldnames)

	def test_get_us_cities(self):
		"""Test get_us_cities"""
		expected = various_tools.get_us_cities(testing=True)
		num_of_cities = 10000
		self.assertLessEqual(num_of_cities, expected)

	def test_split_hyperparameters(self):
		"""Test split_hyperparameters"""
		param = various_tools_fixture.get_hyperparameters()
		boost_vectors, boost_labels, other_vectors = various_tools.split_hyperparameters(param)
		self.assertEqual(boost_vectors, various_tools_fixture.get_boost())
		self.assertEqual(boost_labels, various_tools_fixture.get_boost_labels())
		self.assertEqual(other_vectors, various_tools_fixture.get_non_boost())

if __name__ == '__main__':
	unittest.main()
