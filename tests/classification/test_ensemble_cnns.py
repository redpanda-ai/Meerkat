"""Unit test for meerkat.classification.ensemble_cnn"""

import unittest
import tensorflow as tf
import numpy as np
import meerkat.classification.ensemble_cnns as ensemble_cnns
from nose_parameterized import parameterized

class EnsembleCnnTest(unittest.TestCase):
	"""Unit tests for meerkat.classification.ensemble_cnn"""

	@parameterized.expand([
		(np.array([0.1, 0.2, 0.7]), 2, np.array([0.29, 0.31, 0.4]))
	])
	def test_softmax_with_temperature(self, tensor, temperature, expected):
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
			result = np.round(ensemble_cnns.softmax_with_temperature(tensor, temperature).eval(), decimals=2)
		np.testing.assert_array_equal(result, expected)

	@parameterized.expand([
		(np.array([0.29, 0.31, 0.4]), "logsoftmax", np.array([-1.24, -1.17, -0.92])),
		(np.array([0.29, -1, 0.4]), "logsoftmax",  np.array([-1.24, -23.03, -0.92]))
	])
	def test_logsoftmax(self, softmax, name, expected):
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
			result = np.round(ensemble_cnns.logsoftmax(softmax, name).eval(), decimals=2)
		np.testing.assert_array_equal(result, expected)
