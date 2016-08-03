"""Unit test for meerkat/longtail_handler/bilstm_tagger.py"""

import unittest
import numpy as np
from nose_parameterized import parameterized

from meerkat.longtail import bilstm_tagger as bilstm

class BilstmTaggerTests(unittest.TestCase):
	"""Unittest class for bilstm_tagger"""

	@parameterized.expand([
		(["Debit PIN Purchase ISLAND OF GOLD SUPERMARFRESH MEADOWSNY", "ISLAND OF GOLD SUPERMAR"],
			["background", "background", "background", "merchant", "merchant", "merchant", "merchant", "background"]),
		(["76", "76"],  ["merchant", "merchant"]),
		(["PILOT", "PILOT"], ["merchant", "merchant"]),
		(["PAYMENT THANK YOU", ""], ["background", "background", "background"]),
		(["123 THAI FOOD OAK          HARBOR WA~~08888~~120123052189~~77132~~0~~~0079", "123 THAI FOOD"],
			["merchant", "merchant", "merchant", "background", "background", "background"]),
		(["A.A. Colony Square H.O.A Bill Payment", "A.A. Colony Square"],
			["merchant", "merchant", "merchant", "background", "background", "background"]),
		(["COX CABLE        ONLINE PMT ***********6POS", "COX CABLE"],
			["merchant", "merchant", "background", "background", "background"])
	])
	def test_get_tags(self, description, expected_tags):
		tokens, tags = bilstm.get_tags(description)
		self.assertEqual(tags, expected_tags)

	@parameterized.expand([
		({"tag_map":{"0": "background", "1": "merchant"}}, ["merchant", "background", "merchant"], [[0,1],[1,0],[0,1]]),
		({"tag_map":{"0": "background", "1": "merchant"}}, ["merchant"], [[0,1]])
	])
	def test_encode_tags(self, config, tags, expected):
		result = bilstm.encode_tags(config, tags)
		np.testing.assert_array_equal(list(result), expected)
