"""Unit test for tools files in meerkat/tools"""

from meerkat.tools import make_pulses
from pprint import pformat
import unittest

class VariousToolsTests(unittest.TestCase):

	"""Our UnitTest class."""
	
	expected = """[[{'amount': '151.23',
   'date': '2014-04-01',
   'description': 'Safeway #12, 742 El Camino Real, Sunnyvale CA',
   'ledger_entry': 'credit',
   'transaction_id': '1'},
  {'amount': '100.00',
   'date': '2014-04-03',
   'description': 'Bank Transfer',
   'ledger_entry': 'debit',
   'transaction_id': '2'},
  {'amount': '47.99',
   'date': '2014-04-03',
   'description': 'Amazon.com',
   'ledger_entry': 'credit',
   'transaction_id': '3'}]]"""
	def test_make_pulses(self):
		"""make_pulse test"""

		pulses = make_pulses.split_sample("data/input/make_pulse_test.txt")
		pulses =pformat(pulses)
		self.assertEqual(pulses,self.expected)

if __name__ == '__main__':
	unittest.main()
