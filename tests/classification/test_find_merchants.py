"""
Unit Tests for meerkat.classification.find_merchants

Created on July 1, 2015
@author: Sivan Mehta

Usage:
python3 -m unittest -v tests.classification.test_find_merchants
"""
import unittest

import meerkat.classification.find_merchants as finder

class VariousToolsTests(unittest.TestCase):
	"""Our UnitTest class."""

	def test_find_merchant(self):
		"""find_merchant test to find merchants in strings"""
		transactions = [ \
			"APL*APPLE ITUNES STORE   866-712-7753 CA", \
			"USAA FUNDS TRANSFER CR", \
			"CANTEEN VENDING          8668995849   PA", \
			"SAFEWAY  FUEL 10026409   PUYALLUP     WA", \
			"KOHLS DEPT STORE 420LAKE ZURICH 042006728   8475509400", \
			"someweirdo shenanigans that no one cares about",
			"hahahhahahahahahahahahahahahahahhahahahWALMART" ]
		expected = [ \
			"APPLE",
			"USAA", \
			"CANTEENVENDING", \
			"SAFEWAY", \
			"KOHLS", \
			None,
			"WALMART" ]
		result = []
		for transaction in transactions:
			result.append(finder.find_merchant(transaction))
		self.assertEqual(expected, result)


if __name__ == "__main__":
	unittest.main()
