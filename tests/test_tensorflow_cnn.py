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

if __name__ == '__main__':
	unittest.main()
