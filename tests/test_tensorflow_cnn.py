"""Unit test for meerkat.classification.tensorflow_cnn"""

import unittest
import numpy as np
import meerkat.classification.tensorflow_cnn as tf_cnn

from tests.fixture import tf_cnn_fixture
from nose_parameterized import parameterized

class TensorflowCNNTests(unittest.TestCase):
	"""Our UnitTest class."""

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
	def test_chunks__correct_number(self, array_size, chunk_size):
		"""Test the chunks function to confirm the correct number of chunks"""
		array = np.ones(array_size)
		expected = array_size / chunk_size
		result = len(tf_cnn.chunks(array, chunk_size))
		self.assertEqual(result, expected)

	@parameterized.expand([
		([100, 10]),
		([10, 2])
	])
	def test_chunks_correct_number(self, array_size, chunk_size):
		"""Test the chunks function to confirm the correct size of the chunks"""
		array = np.ones(array_size)
		expected = np.ones(chunk_size)
		results = tf_cnn.chunks(array, chunk_size)
		for chunk in results:
			self.assertTrue(np.allclose(chunk, expected, rtol=1e-05, atol=1e-08))

if __name__ == '__main__':
	unittest.main()
