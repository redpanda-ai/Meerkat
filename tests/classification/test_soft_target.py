"""Unit test for meerkat/classification/soft_target.py"""

import os
import unittest
import meerkat.classification.soft_target as soft_target

from tests.classification.fixture import soft_target_fixture as fixture

class SoftTargetTests(unittest.TestCase):
	"""Unit tests for meerkat.classification.soft_target."""

	def test_get_soft_target(self):
		file_path = soft_target.get_soft_target(fixture.get_data(), fixture.get_models() , "./")
		self.assertEqual(file_path, "./soft_target.csv")

	def tearDown(self):
		try:
			os.remove("./soft_target.csv")
		except OSError:
			pass


