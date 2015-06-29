"""Unit tests for scaling meerkat.location
Created on Jun 27, 2015
@author: J. Andrew Key

"""
import unittest

import meerkat.classification.bloom_filter.find_entities as finder

class VariousToolsTests(unittest.TestCase):
	"""Our UnitTest class."""

	def test_location_split__no_commas(self):
		"""location_split test that it finds San Francisco, CA when there
		are no commas"""
		my_text = "CANDLESTICK PARK SAN FRANCISCO CA SHIRT"
		expect = ('SANFRANCISCO', 'CA')
		result = finder.location_split(my_text)
		self.assertEqual(expect, result)

	def test_location_split__no_answer_is_none(self):
		"""location_split test that the finder does not match
		a location that does not exist"""
		my_text = "CANDLESTICK PARK FAKE CITY CA SHIRT"
		expect = None
		result = finder.location_split(my_text)
		self.assertEqual(expect, result)

	def test_location_split__ignore_city_themed_merchant_names(self):
		"""location_split test that merchant names containing city names
		won't confuse the finder."""
		my_text = "CHICAGO PIZZA SAN JOSE CA"
		expected = ('SANJOSE', 'CA')
		result = finder.location_split(my_text)
		self.assertEqual(expected, result)

	def test_location_split__with_a_comma(self):
		"""location_split test that it finds San Francisco, CA when there
		is a comma"""
		my_text = "CANDLESTICK PARK SAN FRANCISCO, CA SHIRT"
		expected = ('SANFRANCISCO', 'CA')
		result = finder.location_split(my_text)
		self.assertEqual(expected, result)

	def test_location_split_with_punctuation(self):
		"""location_split test that find SF, CA regardless of punctuation"""
		my_text = "Chicago Illumination Company! Located in Davenport, IA?"
		expected = ("CHICAGO", "IL")
		result = finder.location_split(my_text)
		self.assertEqual(expected, result)

	def test_location_split_without_spaces(self):
		"""location_split test to find new york regardless of spaces"""
		my_text = "New York, NY is the most populous city in the US"
		expected = ("NEWYORK", "NY")
		result = finder.location_split(my_text)
		self.assertEqual(expected, result)

	def test_location_split_finding_subset(self):
		"""location_split test to find york, ny in a string that contains
		new york elsewhere"""
		my_text = "A New Yorker president was born in York, NY"
		expected = ("YORK", "NY")
		result = finder.location_split(my_text)
		self.assertEqual(expected, result)

	def test_location_split_smushed(self):
		my_text = "NewYork,NYisthemostpopulouscityintheUS"
		expected = ("NEWYORK", "NY")
		result = finder.location_split(my_text)
		self.assertEqual(expected, result)

if __name__ == "__main__":
	unittest.main()
	sys.exit()
