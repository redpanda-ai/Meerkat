"""Unit test for meerkat.labeling_tools.transaction_labeler"""

import unittest
import meerkat.labeling_tools.transaction_labeler as tl
from nose_parameterized import parameterized

class TransactionLabelerTests(unittest.TestCase):
	"""Unit test for transaction_labeler.py"""

	@parameterized.expand([
		(['20160615_MPANEL_BANK.txt.gz', 'bank']),
		(['20160615_MPANEL_CARD.txt.gz', 'card'])
	])
	def test_identify_container(self, filename, expected_container):
		"""Test for identify_container"""
		self.assertEqual(tl.identify_container(filename), expected_container)

if __name__ == '__main__':
	unittest.main()
