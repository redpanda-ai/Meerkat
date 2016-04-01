"""Unit test for meerkat/classification/tools.py"""

import unittest
import meerkat.classification.tools as tools
from nose_parameterized import parameterized
from tests.fixture import tools_fixture

class ToolsTests(unittest.TestCase):
	"""Our UnitTest class."""

	@parameterized.expand([
		(["missing_input", tools_fixture.get_s3params("missing_input"),
			tools_fixture.get_result("missing_input")]),
		(["missing_slosh", tools_fixture.get_s3params("missing_slosh"),
			tools_fixture.get_result("missing_slosh")]),
		(["unpreprocessed", tools_fixture.get_s3params("unpreprocessed"),
			tools_fixture.get_result("unpreprocessed")]),
		(["preprocessed", tools_fixture.get_s3params("preprocessed"),
			tools_fixture.get_result("preprocessed")])
	])
	def test_check_new_input_file(self, case_type, s3params, result):
		"""Test check_new_input_file"""
		if case_type == "missing_input":
			self.assertRaises(SystemExit, tools.check_new_input_file, **s3params)
		else:
			self.assertEqual(tools.check_new_input_file(**s3params), result)

