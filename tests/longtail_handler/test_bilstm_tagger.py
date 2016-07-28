"""Unit test for meerkat/longtail_handler/bilstm_tagger.py"""

import unittest
from nose_parameterized import parameterized

from meerkat.longtail_handler import bilstm_tagger as bilstm

class MioTests(unittest.TestCase):
	"""Unittest class for bilstm_tagger"""

	@parameterized.expand([
		(["Debit PIN Purchase ISLAND OF GOLD SUPERMARFRESH MEADOWSNY", "ISLAND OF GOLD SUPERMAR"],
			["background", "background", "background", "merchant", "merchant", "merchant", "merchant", "background"]),
		(["76", "76"],  ["merchant"]),
		(["PAYMENT THANK YOU", ""], ["background", "background", "background"]),
		(["123 THAI FOOD OAK          HARBOR WA~~08888~~120123052189~~77132~~0~~~0079", "123 THAI FOOD"],
			["merchant", "merchant", "merchant", "background", "background", "background"]),
		(["A.A. Colony Square H.O.A Bill Payment", "A.A. Colony Square"],
			["merchant", "merchant", "merchant", "background", "background", "background"])
	])
	def test_get_tags(self, description, expected_tags):
		tokens, tags = bilstm.get_tags(description)
		self.assertEqual(tags, expected_tags)
