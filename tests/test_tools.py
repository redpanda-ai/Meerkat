"""Unit test for tools files in meerkat/tools"""

from meerkat.tools import make_pulses
from pprint import pformat
import unittest

class VariousToolsTests(unittest.TestCase):

	"""Our UnitTest class."""
	
	expected = """[[{'transaction_id': '1',
			 'description': 'Safeway #12, 742 El Camino Real, Sunnyvale CA 94087',
			 'amount': '151.23',
			 'ledger_entry': 'credit',
			 'date': '2014-04-01'},
			{'transaction_id': '2',
			 'description': 'Bank Transfer',
			 'amount': '100.00',
			 'ledger_entry': 'debit',
			 'date': '2014-04-03'},
			{'transaction_id': '3',
			 'description': 'Amazon.com',
			 'amount': '47.99',
			 'ledger_entry': 'credit',
			 'date': '2014-04-03'}]]"""


	def test_make_pulses(self):
		"""make_pulse test"""

		pulses = split_sample("/data/input/make_pulse_test.txt")
		pulses = pformat(pulses)
		self.assertEqual(pulses,expected)

if __name__ == '__main__':
	unittest.main()
