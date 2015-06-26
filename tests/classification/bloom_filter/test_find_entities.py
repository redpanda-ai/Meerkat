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
		expect = ('SAN FRANCISCO', 'CA')
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
		expect = ('SAN JOSE', 'CA')
		result = finder.location_split(my_text)
		self.assertEqual(expect, result)

	def test_location_split__with_a_comma(self):
		"""location_split test that it finds San Francisco, CA when there
		is a comma"""
		my_text = "CANDLESTICK PARK SAN FRANCISCO, CA SHIRT"
		expect = ('SAN FRANCISCO', 'CA')
		result = finder.location_split(my_text)
		self.assertEqual(expect, result)

	def test_location_split_with_punctuation(self):
		"""location_split test that find SF, CA regardless of punctuation"""
		my_text = "Chicago Illumination Company! Located in Davenport, IA?"
		expect = ("DAVENPORT", "IA")
		result = finder.location_split(my_text)
		self.assertEqual(expect, result)

if __name__ == "__main__":
	unittest.main()
	sys.exit()
