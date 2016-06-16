"""Unit test for meerkat.labeling_tools.get_merchants_by_id"""

import sys
import unittest
import meerkat.labeling_tools.get_merchants_by_id as getid
from nose_parameterized import parameterized
from tests.labeling_tools.fixture import get_merchants_by_id_fixture

class GetMerchantsByIdTests(unittest.TestCase):
	"""Unit tests for get_merchants_by_id"""

	@parameterized.expand([
		(["not_enough", get_merchants_by_id_fixture.get_args()["not_enough"]]),
		(["no_json", get_merchants_by_id_fixture.get_args()["no_json"]]),
		(["no_txt", get_merchants_by_id_fixture.get_args()["no_txt"]]),
		(["not_correct", get_merchants_by_id_fixture.get_args()["not_correct"]]),
		(["correct", get_merchants_by_id_fixture.get_args()["correct"]])
	])
	def test_verify_arguments(self, case, args):
		"""Test verify_arguments with parameters"""
		sys.argv = args
		if case == "not_enough" or case == "no_json" or case == "no_txt" or case == "not_correct":
			self.assertRaises(SystemExit, getid.verify_arguments)
		else:
			getid.verify_arguments()

if __name__ == '__main__':
	unittest.main()
