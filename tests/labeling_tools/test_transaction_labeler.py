"""Unit test for meerkat.labeling_tools.transaction_labeler"""

import sys
import unittest
from nose_parameterized import parameterized
import meerkat.labeling_tools.transaction_labeler as tl
from tests.labeling_tools.fixture import transaction_labeler_fixture

class TransactionLabelerTests(unittest.TestCase):
	"""Unit test for transaction_labeler.py"""

	@parameterized.expand([
		(['20160615_MPANEL_BANK.txt.gz', 'bank']),
		(['20160615_MPANEL_CARD.txt.gz', 'card'])
	])
	def test_identify_container(self, filename, expected_container):
		"""Test for identify_container"""
		self.assertEqual(tl.identify_container(filename), expected_container)

	@parameterized.expand([
		(["not_enough", transaction_labeler_fixture.get_args()["not_enough"]]),
		(["enough", transaction_labeler_fixture.get_args()["enough"]])
	])
	def test_verify_arguments(self, case, args):
		"""Test verify_arguments with parameters"""
		sys.argv = args
		if case == "not_enough":
			self.assertRaises(SystemExit, tl.verify_arguments)
		else:
			tl.verify_arguments()

if __name__ == '__main__':
	unittest.main()
