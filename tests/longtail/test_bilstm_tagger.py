"""Unit test for meerkat/longtail_handler/bilstm_tagger.py"""

import unittest
import numpy as np
from nose_parameterized import parameterized

from meerkat.longtail import bilstm_tagger as bilstm

class BilstmTaggerTests(unittest.TestCase):
	"""Unittest class for bilstm_tagger"""

	@parameterized.expand([
		({"Description": "Debit PIN Purchase ISLAND OF GOLD SUPERMARFRESH MEADOWSNY", "Tagged_merchant_string": "ISLAND OF GOLD SUPERMAR"},
			["background", "background", "background", "merchant", "merchant", "merchant", "merchant", "background"]),
		({"Description": "76", "Tagged_merchant_string": "76"},  ["merchant"]),
		({"Description": "PAYMENT THANK YOU", "Tagged_merchant_string": ""}, ["background", "background", "background"]),
		({"Description": "123 THAI FOOD OAK          HARBOR WA~~08888~~120123052189~~77132~~0~~~0079", "Tagged_merchant_string": "123 THAI FOOD"},
			["merchant", "merchant", "merchant", "background", "background", "background"]),
		({"Description": "COX CABLE        ONLINE PMT ***********6POS", "Tagged_merchant_string": "COX CABLE"},
			["merchant", "merchant", "background", "background", "background"]),
		({"Description": "AMERICAN EXPRESS DES:SETTLEMENT ID:5049791080                INDN:SUBWAY #29955049791080  CO ID:1134992250 CCD", "Tagged_merchant_string": "AMERICAN EXPRESS, SUBWAY"},
			["merchant", "merchant", "background", "background", "merchant", "background", "background", "background", "background"]),
		({"Description": "AA MILES BY POINTS     POINTS.COM    IL", "Tagged_merchant_string": "AA, Points.com"},
			["merchant", "background", "background", "background", "merchant", "background"])
	])
	def test_get_tags(self, description, expected_tags):
		config = {"max_tokens": 35}
		tokens, tags = bilstm.get_tags(config, description)
		self.assertEqual(tags, expected_tags)

	@parameterized.expand([
		({"tag_map":{"0": "background", "1": "merchant"}}, ["merchant", "background", "merchant"], [[0,1],[1,0],[0,1]]),
		({"tag_map":{"0": "background", "1": "merchant"}}, ["merchant"], [[0,1]])
	])
	def test_encode_tags(self, config, tags, expected):
		result = bilstm.encode_tags(config, tags)
		np.testing.assert_array_equal(list(result), expected)
