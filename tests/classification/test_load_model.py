"""Unit test for meerkat/classification/load_model"""

import unittest
import meerkat.classification.load_model as load_model

from nose_parameterized import parameterized, param
from tests.classification.fixture import load_model_fixture as fixture

class LoadModelTests(unittest.TestCase):
	"""UnitTest class."""

	@parameterized.expand([
		param(fixture.get_trans(), None, doc_key="DESCRIPTION_UNMASKED"),
		param(fixture.get_trans(), fixture.get_class_size(), doc_key="DESCRIPTION_UNMASKED", soft_target=True),
		param(fixture.get_trans(), None)
	])
	def test_apply_cnn(self, trans, expected, doc_key="description", soft_target=False):
		"""Tests if a classifier returns value as expected"""
		apply_cnn = load_model.get_tf_cnn_by_name("bank_debit_subtype", gpu_mem_fraction=True)
		if doc_key=="DESCRIPTION_UNMASKED":
			result = apply_cnn(trans, doc_key=doc_key, soft_target=soft_target)[0]
			if soft_target:
				self.assertEqual(len(result), expected)
			else:
				self.assertTrue(isinstance(result["CNN"], str))
		else:
			with self.assertRaises(KeyError):
				apply_cnn(trans)
