"""Unit tests for meerkat.classification.bloom_filter.trie"""

import sys
import unittest

import meerkat.classification.bloom_filter.trie as trie

class TrieTests(unittest.TestCase):
	"""Our UnitTest class."""

	def test_location_split__no_commas(self):
		"""location_split test that it finds San Francisco, CA when there
		are no commas"""
		my_text = "CANDLESTICK PARK SAN FRANCISCO CA SHIRT"
		expect = ('San Francisco', 'CA')
		result = trie.location_split(my_text)
		self.assertEqual(expect, result)

	def test_location_split__no_answer_is_none(self):
		"""location_split test that the finder does not match
		a location that does not exist"""
		my_text = "CANDLESTICK PARK FAKE CITY CA SHIRT"
		expect = None
		result = trie.location_split(my_text)
		self.assertEqual(expect, result)

	def test_location_split__ignore_city_themed_merchant_names(self):
		"""location_split test that merchant names containing city names
		won't confuse the finder."""
		my_text = "CHICAGO PIZZA SAN JOSE CA"
		expected = ('San Jose', 'CA')
		result = trie.location_split(my_text)
		self.assertEqual(expected, result)

	def test_location_split__with_a_comma(self):
		"""location_split test that it finds San Francisco, CA when there
		is a comma"""
		my_text = "CANDLESTICK PARK SAN FRANCISCO, CA SHIRT"
		expected = ('San Francisco', 'CA')
		result = trie.location_split(my_text)
		self.assertEqual(expected, result)

	def test_location_split_with_punctuation(self):
		"""location_split test that find SF, CA regardless of punctuation"""
		my_text = "Chicago Illumination Company! Located in Davenport, IA?"
		expected = ("Davenport", "IA")
		result = trie.location_split(my_text)
		self.assertEqual(expected, result)

	def test_location_split_without_spaces(self):
		"""location_split test to find new york regardless of spaces"""
		my_text = "New York, NY is the most populous city in the US"
		expected = ("New York", "NY")
		result = trie.location_split(my_text)
		self.assertEqual(expected, result)

	def test_location_split_finding_subset(self):
		"""location_split test to find york, ny in a string that contains
		new york elsewhere"""
		my_text = "A New Yorker president was born in York, NY"
		expected = ("York", "NY")
		result = trie.location_split(my_text)
		self.assertEqual(expected, result)

	def test_location_split_nyc_to_nyc(self):
		"""location_split test to find york, ny in a string that contains
		new york elsewhere"""
		my_text = "Andy bought a car in New York City, NY!!"
		expected = ("New York City", "NY")
		result = trie.location_split(my_text)
		self.assertEqual(expected, result)

	def test_location_split_union_city_to_union_city(self):
		"""location_split test to find york, ny in a string that contains
		new york elsewhere"""
		my_text = "Andy bought another car in Union City, NJ!!"
		expected = ("Union City", "NJ")
		result = trie.location_split(my_text)
		self.assertEqual(expected, result)

#	def test_location_split_union_to_union(self):
#		"""location_split test to find york, ny in a string that contains
#		new york elsewhere"""
#		my_text = "Andy bought yet another car in Union, NJ!!"
#		expected = ("Union", "NJ")
#		result = trie.location_split(my_text)
#		self.assertEqual(expected, result)

	def test_location_split_smushed(self):
		"""location_split test to find new york ny in string that is 
		'smushed' to be without spaces"""
		my_text = "NewYork,NYisthemostpopulouscityintheUS"
		expected = ("New York", "NY")
		result = trie.location_split(my_text)
		self.assertEqual(expected, result)

	def test_location_chase_la(self):
		"""location_split test to find irving texas where 
		chase la is clearly in there"""
		my_text = "Debit Card Purchase LA MICHOACANA # 26 IRVING TX"
		expected = ("Irving", "TX")
		result = trie.location_split(my_text)
		self.assertEqual(expected, result)

if __name__ == "__main__":
	unittest.main()
	sys.exit()
