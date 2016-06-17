"""Unit test for meerkat.classification.tensorflow_cnn"""

import unittest
import pandas as pd
import meerkat.classification.tensorflow_cnn as tf_cnn
import jsonschema as js
import numpy as np
from meerkat.various_tools import load_params
from tests.classification.fixture import tf_cnn_fixture
from nose_parameterized import parameterized

class TfcnnTest(unittest.TestCase):
	"""Unit tests for meerkat.classification.tensorflow_cnn"""

	@parameterized.expand([
		([tf_cnn_fixture.get_config_for_batch_to_tensor(), tf_cnn_fixture.get_batch(),
			tf_cnn_fixture.get_trans_and_labels()])
	])
	def test_batch_to_tensor(self, config, batch, expected):
		"""Test batch_to_tensor with parameters"""
		result = tf_cnn.batch_to_tensor(config, batch)
		self.assertTrue((result[0]==expected[0]).all())
		self.assertTrue((result[1]==expected[1]).all())

	@parameterized.expand([
		([['invalid', 'tests/classification/fixture/missing_entry_tf_config.json']]),
		([['valid', 'tests/classification/fixture/valid_tf_config.json']]),
		([['invalid', 'tests/classification/fixture/wrong_doc_length_tf_config.json']])
	])
	def test_validate_config(self, config):
		"""Confirm that the validate_config functions as expected."""
		case, config = config[:]
		if case == 'invalid':
			self.assertRaises(js.exceptions.ValidationError, tf_cnn.validate_config, config)
		elif case == 'valid':
			self.assertEqual(type(tf_cnn.validate_config(config)), dict)

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
