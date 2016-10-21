"""Unit test for meerkat/longtail_handler/bilstm_tagger.py"""

import unittest
import numpy as np
import pandas as pd
from nose_parameterized import parameterized

from meerkat.longtail import bilstm_tagger as bilstm
from tests.longtail.fixture import bilstm_tagger_fixture

class BilstmTaggerTests(unittest.TestCase):
	"""Unittest class for bilstm_tagger"""

	def test_tokenize(self):
		"""Test is transactions are properly tokenized"""

		input_trans = [
			"1227WENDY'S VALDOSTA GA                  0XXXXXXXXXXXXXXXX",
			"9668491207Staples, Inc BURTON MI               0XXXXXXXXXXXXXXXX",
			"1218BP#9493560CIRCLE K 2708 ORLANDO FL   0XXXXXXXXXXXXXXXX",
			"5288941207TARGET T2760 TARGET T2 OXNARD CA     0XXXXXXXXXXXXXXXX",
			"1223WM SUPERC Wal-Mart Sup PALMDALE CA   0XXXXXXXXXXXXXXXX",
			"1210AmazonPrime Membersh amzn.com/prme NV0XXXXXXXXXXXXXXXX",
			"123 THAI FOOD OAK          HARBOR WA~~08888~~120123052189~~77132~~0~~~0079"
		]

		output_trans = [
			"1227 WENDY'S VALDOSTA GA 0 XXXXXXXXXXXXXXXX",
			"9668491207 Staples, Inc BURTON MI 0 XXXXXXXXXXXXXXXX",
			"1218 BP #9493560 CIRCLE K 2708 ORLANDO FL 0 XXXXXXXXXXXXXXXX",
			"5288941207 TARGET T 2760 TARGET T 2 OXNARD CA 0 XXXXXXXXXXXXXXXX",
			"1223 WM SUPERC Wal-Mart Sup PALMDALE CA 0 XXXXXXXXXXXXXXXX",
			"1210 AmazonPrime Membersh amzn.com/prme NV 0 XXXXXXXXXXXXXXXX",
			"123 THAI FOOD OAK HARBOR WA ~~08888~~120123052189~~77132~~0~~~0079"
		]

		for key, value in zip(input_trans, output_trans):
			self.assertEqual(bilstm.tokenize(key), value.split())

	def test_get_token_tag_pairs(self):
		"""test if get_tags produces tag for each token correctly"""

		config = bilstm_tagger_fixture.get_config()
		transactions = bilstm_tagger_fixture.get_token_tag_pairs_input()

		for trans in transactions:
			tokens, tags = bilstm.get_token_tag_pairs(config, trans[0])
			self.assertEqual(tags, trans[1])
			self.assertEqual(tokens, trans[2])

	@parameterized.expand([
		({"tag_map": {"0": "background", "1": "merchant"}}, ["merchant", "background", "merchant"], [[0,1],[1,0],[0,1]]),
		({"tag_map": {"0": "background", "1": "merchant"}}, ["merchant"], [[0,1]])
	])
	def test_encode_tags(self, config, tags, expected):
		"""test one-hot encoding the tags"""
		result = bilstm.encode_tags(config, tags)
		np.testing.assert_array_equal(list(result), expected)

	def test_validate_config(self):
		"""make sure special chars have indices"""
		fixture_config = bilstm_tagger_fixture.get_config()
		result = bilstm.validate_config(fixture_config)
		self.assertEqual(result["c2i"]["_UNK"],  0)
		self.assertEqual(result["c2i"]["<w>"], 1)
		self.assertEqual(result["c2i"]["</w>"], 2)

	def test_words_to_indices(self):
		"""test words and indices are 1 to 1 relation"""
		fixture = [(["amazon", "prime", "purchase", "amazon"], ["merchant", "merchant", "background", "merchant"])]
		result = bilstm.words_to_indices(fixture)
		self.assertEqual(len(result.keys()), 4)
		self.assertEqual(result["_UNK"], 0)

	@parameterized.expand([
		({}, 4),
		({"abc": [1, 1, 2], "supermarket": [4,5,6]}, 6),
		({"amazon": [1, 1, 2], "supermarket": [4,5,6]}, 5)
	])
	def test_construct_embedding(self, loaded_embedding, expected_w2i_length):
		"""test word imbedding is properly constructed"""
		fixture_config = {"we_dim": 3}
		fixture_w2i = {"_UNK": 0, "amazon": 1, "prime": 2, "purchase": 3}
		result_w2i, result_wembedding = bilstm.construct_embedding(fixture_config, fixture_w2i, loaded_embedding)
		self.assertEqual(len(result_w2i.keys()), expected_w2i_length)
		self.assertEqual(result_wembedding.shape, (expected_w2i_length, fixture_config["we_dim"]))


	@parameterized.expand([
		(["76"], ["merchant"], [[1, 36, 35 ,2], [0, 0, 0, 0]], [4, 0], [0]),
		(["ama", "purchase"], None, [[1, 3, 15 ,3, 2, 0, 0, 0, 0, 0],[1, 18, 23, 20, 5, 10, 3, 21, 7, 2]], [5, 10], [0, 3])
	])
	def test_trans_to_tensor(self, tokens, tags, exp_char_inputs, exp_word_length, exp_word_indices):
		"""test if tokens and chars got translated to indices properly"""
		fixture_config = {
			"alphabet": "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
			"tag_column_map": {
				"MERCHANT_NAME": "merchant",
				"LOCALITY": "city",
				"STORE_NUMBER": "store_number",
				"PHONE_NUMBER": "phone_number",
				"STATE": "state"
			},
			"tag_map": {
				"0": "background",
				"1": "merchant",
				"2": "store_number",
				"3": "city",
				"4": "state",
				"5": "phone_number"
			},
		}
		fixture_config = bilstm.validate_config(fixture_config)
		fixture_config["w2i"] = {"_UNK": 0, "amazon": 1, "prime": 2, "purchase": 3}
		fixture_config["max_tokens"] = 2
		char_inputs, word_lengths, word_indices, encoded_tags = bilstm.trans_to_tensor(fixture_config, tokens, tags=tags)
		np.testing.assert_array_equal(list(np.transpose(char_inputs, (1,0))), exp_char_inputs)
		self.assertEqual(word_lengths, exp_word_length)
		self.assertEqual(word_indices, exp_word_indices)
