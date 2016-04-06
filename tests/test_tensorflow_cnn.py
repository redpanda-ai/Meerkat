"""Unit test for meerkat.classification.tensorflow_cnn"""

import unittest
import csv
import pandas as pd
import meerkat.classification.tensorflow_cnn as tf_cnn
import jsonschema as js
import numpy as np
from meerkat.various_tools import load_params
from tests.fixture import tf_cnn_fixture
from nose_parameterized import parameterized

class TfcnnTest(unittest.TestCase):
	"""Unit tests for meerkat.classification.tensorflow_cnn"""
	@parameterized.expand([
		([[], 3, []]),
		([[1], 3, [[1]]]),
		([[1,2,3], 2, [[1,2],[3]]]),
		([[1,2], -9, [[1],[2]]])
	])
	def test_chunks(self, array, num, expected):
		"""Ensure that numpy arrays are properly sub-divided into chunks."""
		self.assertEqual(tf_cnn.chunks(array, num), expected)

	@parameterized.expand([
		([['invalid', 'tests/fixture/missing_entry_tf_config.json']]),
		([['valid', 'tests/fixture/valid_tf_config.json']]),
		([['invalid', 'tests/fixture/wrong_doc_length_tf_config.json']])
	])
	def test_validate_config(self, config):
		"""Confirm that the validate_config functions as expected."""
		case, config = config[:]
		if case == 'invalid':
			self.assertRaises(js.exceptions.ValidationError, tf_cnn.validate_config, config)
		elif case == 'valid':
			self.assertEqual(type(tf_cnn.validate_config(config)), dict)

	@parameterized.expand([
		(['tests/fixture/tf_config_with_alpha_dict.json', 'kkk', 3, [[0,0,0],[0,0,0],[0,0,0]]]),
		(['tests/fixture/tf_config_with_alpha_dict.json', 'abc', 3, [[0,0,1],[0,1,0],[1,0,0]]]),
		(['tests/fixture/tf_config_with_alpha_dict.json', 'abccccccccc', 3, [[0,0,1],[0,1,0],[1,0,0]]]),
		(['tests/fixture/tf_config_with_alpha_dict.json', 'a', 3, [[1,0,0],[0,0,0],[0,0,0]]])
	])
	def test_string_to_tensor(self, config, doc, length, expected):
		config = load_params(config)
		np.testing.assert_array_equal(list(tf_cnn.string_to_tensor(config, doc, length)), expected)

	@parameterized.expand([
		([tf_cnn_fixture.get_predictions("all_correct"), tf_cnn_fixture.get_labels(), 100.0]),
		([tf_cnn_fixture.get_predictions("all_wrong"), tf_cnn_fixture.get_labels(), 0.0]),
		([tf_cnn_fixture.get_predictions("half_correct"), tf_cnn_fixture.get_labels(), 50.0])
	])
	def test_accuracy(self, predictions, labels, expected_accuracy):
		"""Test accuracy with parameters"""
		result = tf_cnn.accuracy(predictions, labels)
		self.assertEqual(result, expected_accuracy)

	@parameterized.expand([
		([tf_cnn_fixture.get_config(), "aab", 4, tf_cnn_fixture.get_tensor("short_doc")]),
		([tf_cnn_fixture.get_config(), "aab", 2, tf_cnn_fixture.get_tensor("truncated_doc")])
	])
	def test_string_to_tensor(self, config, doc, length, expected_tensor):
		"""Test string_to_tensor with parameters"""
		result = tf_cnn.string_to_tensor(config, doc, length)
		np.testing.assert_array_equal(result, expected_tensor)

	@parameterized.expand([
		([100, 10]),
		([10, 2])
	])
	def test_chunks__correct_number_of_chunks(self, array_size, chunk_size):
		"""Test the chunks function to confirm the correct number of chunks"""
		array = np.ones(array_size)
		expected = array_size / chunk_size
		result = len(tf_cnn.chunks(array, chunk_size))
		self.assertEqual(result, expected)

	@parameterized.expand([
		([100, 10]),
		([10, 2])
	])
	def test_chunks__correct_chunk_size(self, array_size, chunk_size):
		"""Test the chunks function to confirm the correct size of the chunks"""
		array = np.ones(array_size)
		expected = np.ones(chunk_size)
		results = tf_cnn.chunks(array, chunk_size)
		for chunk in results:
			self.assertTrue(np.allclose(chunk, expected, rtol=1e-05, atol=1e-08))

	def test_load_data__number_of_labels_exception(self):
		"""Confirm that we throw an informative Exception when the number of labels
		fails to match the number of label_keys in a value count"""
		config = tf_cnn_fixture.get_subtype_config()
		self.assertRaisesRegex(Exception, "Number of indexes does not match number of labels", tf_cnn.load_data, config)

if __name__ == '__main__':
	unittest.main()
